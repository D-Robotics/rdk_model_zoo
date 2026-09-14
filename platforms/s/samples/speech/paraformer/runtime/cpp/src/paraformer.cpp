/**
 * @file paraformer.cpp
 * @brief Paraformer S100 end-to-end UCP/HB-DNN inference implementation.
 *
 * This implementation preserves the validated Encoder -> Predictor -> CPU CIF
 * -> Decoder execution order and evaluates pre-extracted fbank+LFR features
 * from a JSON manifest. It is called by the standard ``main.cpp`` entry point.
 */
#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <nlohmann/json.hpp>

#include "paraformer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;
using clock_ = std::chrono::steady_clock;

/**
 * @brief Return elapsed milliseconds from a steady-clock start point.
 *
 * @param[in] t0 Start point.
 * @return Elapsed milliseconds.
 */
static double ms_since(clock_::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_::now() - t0).count();
}

#define CHECK(expr) do { int32_t _r = (expr); \
    if (_r != 0) { std::cerr << #expr " failed: " << _r << " at line " << __LINE__ << std::endl; std::exit(1); } \
} while (0)

/**
 * @brief Minimal C-contiguous little-endian NumPy array representation.
 */
struct NpyArray {
    std::vector<int64_t> shape;
    std::string dtype;   // "<f4", "<i4", "<i8"
    std::vector<uint8_t> data;
    size_t elem_size() const {
        if (dtype == "<f4" || dtype == "<i4") return 4;
        if (dtype == "<i8") return 8;
        if (dtype == "<f2") return 2;
        return 0;
    }
    size_t num_elements() const {
        size_t n = 1;
        for (auto d : shape) n *= (size_t)d;
        return n;
    }
};

/**
 * @brief Load a supported NumPy ``.npy`` array from disk.
 *
 * @param[in] path Input feature path.
 * @return Parsed array metadata and raw data.
 */
static NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << std::endl; std::exit(1); }
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "not npy: " << path << std::endl; std::exit(1);
    }
    uint8_t major, minor;
    f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t header_len = 0;
    if (major == 1) { uint16_t h; f.read((char*)&h, 2); header_len = h; }
    else            { f.read((char*)&header_len, 4); }
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    NpyArray a;
    // Parse: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 400, 560), }
    auto get_val = [&](const std::string& key) -> std::string {
        auto p = header.find("'" + key + "':");
        if (p == std::string::npos) return "";
        auto start = p + key.size() + 3;  // past 'key':
        while (start < header.size() && header[start] == ' ') ++start;
        if (start < header.size() && header[start] == '(') {
            auto q = header.find(')', start);
            return header.substr(start, q - start + 1);
        }
        auto q = header.find_first_of(",}", start);
        return header.substr(start, q - start);
    };
    std::string desc = get_val("descr");
    // strip quotes / spaces
    while (!desc.empty() && (desc.front() == ' ' || desc.front() == '\'')) desc.erase(0, 1);
    while (!desc.empty() && (desc.back()  == ' ' || desc.back()  == '\'')) desc.pop_back();
    a.dtype = desc;
    std::string shp = get_val("shape");
    // strip "()" and spaces
    auto lp = shp.find('('); auto rp = shp.find(')');
    if (lp != std::string::npos) shp = shp.substr(lp + 1, rp - lp - 1);
    size_t pos = 0;
    while (pos < shp.size()) {
        while (pos < shp.size() && (shp[pos] == ' ' || shp[pos] == ',')) ++pos;
        if (pos >= shp.size()) break;
        char* end;
        int64_t v = std::strtoll(shp.c_str() + pos, &end, 10);
        if (end == shp.c_str() + pos) break;
        a.shape.push_back(v);
        pos = end - shp.c_str();
    }
    size_t nb = a.num_elements() * a.elem_size();
    a.data.resize(nb);
    f.read((char*)a.data.data(), nb);
    return a;
}

// ==== hbm model wrapper ====
/**
 * @brief One hbDNN model together with allocated input and output tensors.
 */
struct Model {
    hbDNNHandle_t handle = nullptr;
    std::string   name;
    std::vector<std::string> input_names, output_names;
    std::vector<hbDNNTensorProperties> input_props, output_props;
    std::vector<hbDNNTensor> inputs, outputs;

    void init(hbDNNPackedHandle_t packed, const std::string& mname) {
        name = mname;
        CHECK(hbDNNGetModelHandle(&handle, packed, mname.c_str()));
        int32_t nin, nout;
        CHECK(hbDNNGetInputCount(&nin, handle));
        CHECK(hbDNNGetOutputCount(&nout, handle));
        input_names.resize(nin); output_names.resize(nout);
        input_props.resize(nin); output_props.resize(nout);
        inputs.resize(nin); outputs.resize(nout);
        for (int i = 0; i < nin; ++i) {
            char const* nm; CHECK(hbDNNGetInputName(&nm, handle, i));
            input_names[i] = nm;
            CHECK(hbDNNGetInputTensorProperties(&input_props[i], handle, i));
            inputs[i].properties = input_props[i];
            CHECK(hbUCPMalloc(&inputs[i].sysMem, input_props[i].alignedByteSize, 0));
        }
        for (int i = 0; i < nout; ++i) {
            char const* nm; CHECK(hbDNNGetOutputName(&nm, handle, i));
            output_names[i] = nm;
            CHECK(hbDNNGetOutputTensorProperties(&output_props[i], handle, i));
            outputs[i].properties = output_props[i];
            CHECK(hbUCPMalloc(&outputs[i].sysMem, output_props[i].alignedByteSize, 0));
        }
    }

    void release() {
        for (auto& t : inputs)  if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
        for (auto& t : outputs) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    }

    int input_idx(const std::string& n) const {
        for (size_t i = 0; i < input_names.size(); ++i) if (input_names[i] == n) return (int)i;
        return -1;
    }
    int output_idx(const std::string& n) const {
        for (size_t i = 0; i < output_names.size(); ++i) if (output_names[i] == n) return (int)i;
        return -1;
    }

    // Copy src (compact row-major) → BPU tensor (may be aligned/padded).
    static void copy_tensor(void* dst, const void* src,
                            const hbDNNTensorProperties& p, bool to_bpu) {
        const auto& sh = p.validShape;
        int nd = sh.numDimensions;
        // Element size from tensorType
        int esz = 4; // default float32
        switch (p.tensorType) {
            case HB_DNN_TENSOR_TYPE_U8: case HB_DNN_TENSOR_TYPE_S8: esz = 1; break;
            case HB_DNN_TENSOR_TYPE_F16: case HB_DNN_TENSOR_TYPE_S16:
            case HB_DNN_TENSOR_TYPE_U16: esz = 2; break;
            case HB_DNN_TENSOR_TYPE_F32: case HB_DNN_TENSOR_TYPE_S32:
            case HB_DNN_TENSOR_TYPE_U32: esz = 4; break;
            case HB_DNN_TENSOR_TYPE_F64: case HB_DNN_TENSOR_TYPE_S64:
            case HB_DNN_TENSOR_TYPE_U64: esz = 8; break;
        }
        // Compute natural (compact) stride bytes
        int64_t nat_stride[HB_DNN_TENSOR_MAX_DIMENSIONS];
        nat_stride[nd - 1] = esz;
        for (int d = nd - 2; d >= 0; --d) nat_stride[d] = nat_stride[d + 1] * sh.dimensionSize[d + 1];
        // Fast path: strides match — flat memcpy
        bool match = true;
        for (int d = 0; d < nd; ++d) if (p.stride[d] != nat_stride[d]) { match = false; break; }
        if (match) {
            size_t nb = (size_t)nat_stride[0] * sh.dimensionSize[0];
            if (to_bpu) std::memcpy(dst, src, nb);
            else        std::memcpy(dst, src, nb);
            return;
        }
        // Slow path: iterate last-dim rows and copy inner slab
        // Only handle simple case: last dim stride differs (row padding)
        size_t row_bytes_nat = (size_t)nat_stride[nd - 2];  // natural row size
        size_t row_bytes_aln = (size_t)p.stride[nd - 2];    // aligned row size
        size_t n_rows = 1;
        for (int d = 0; d < nd - 1; ++d) n_rows *= (size_t)sh.dimensionSize[d];
        char* d_ptr = (char*)dst;
        const char* s_ptr = (const char*)src;
        for (size_t r = 0; r < n_rows; ++r) {
            if (to_bpu) {
                std::memcpy(d_ptr + r * row_bytes_aln, s_ptr + r * row_bytes_nat, row_bytes_nat);
            } else {
                std::memcpy(d_ptr + r * row_bytes_nat, s_ptr + r * row_bytes_aln, row_bytes_nat);
            }
        }
    }

    void write_input(int i, const void* src, size_t /*n_bytes*/) {
        copy_tensor(inputs[i].sysMem.virAddr, src, input_props[i], true);
        CHECK(hbUCPMemFlush(&inputs[i].sysMem, HB_SYS_MEM_CACHE_CLEAN));
    }

    void read_output(int i, void* dst, size_t /*n_bytes*/) {
        CHECK(hbUCPMemFlush(&outputs[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE));
        copy_tensor(dst, outputs[i].sysMem.virAddr, output_props[i], false);
    }

    double infer() {
        auto t0 = clock_::now();
        hbUCPTaskHandle_t task = nullptr;
        CHECK(hbDNNInferV2(&task, outputs.data(), inputs.data(), handle));
        hbUCPSchedParam sched;
        HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
        sched.backend = HB_UCP_BPU_CORE_ANY;
        CHECK(hbUCPSubmitTask(task, &sched));
        CHECK(hbUCPWaitTaskDone(task, 0));
        CHECK(hbUCPReleaseTask(task));
        return ms_since(t0);
    }
};

// ==== CIF (CPU, numpy-equivalent) ====
// alphas [1, 401], concat5 [1, 401, 512], real_T (int)
// out: frame_fires [1, 100, 512], token_num [1] int32
constexpr int MAX_LABEL_LEN = 100;
/**
 * @brief Execute the validated CPU Continuous Integrate-and-Fire operation.
 *
 * @param[in] alphas Predictor activation values.
 * @param[in] concat5 Predictor acoustic embeddings.
 * @param[in] real_T Valid feature-frame count before fixed-shape padding.
 * @param[out] frame_fires Decoder acoustic embeddings.
 * @param[out] token_num Decoder token count.
 */
static void cif_numpy(const float* alphas, const float* concat5, int real_T,
                      float* frame_fires_out, int32_t* token_num_out) {
    const int T = 401;
    const int H = 512;
    // Mask alphas beyond real_T
    std::vector<float> alphas_m(T);
    for (int t = 0; t < T; ++t)
        alphas_m[t] = (real_T >= 0 && t >= real_T) ? 0.f : alphas[t];

    std::vector<double> ps(T);
    ps[0] = alphas_m[0];
    for (int t = 1; t < T; ++t) ps[t] = ps[t - 1] + (double)alphas_m[t];
    std::vector<float> prefix_sum(T);
    for (int t = 0; t < T; ++t) prefix_sum[t] = (float)ps[t];

    std::vector<float> psf(T), dpsf(T);
    for (int t = 0; t < T; ++t) psf[t] = std::floor(prefix_sum[t]);
    dpsf[0] = 0.f;
    for (int t = 1; t < T; ++t) dpsf[t] = std::floor(prefix_sum[t - 1]);

    std::vector<uint8_t> fire_idx(T);
    for (int t = 0; t < T; ++t) fire_idx[t] = (psf[t] - dpsf[t]) > 0 ? 1 : 0;

    std::vector<float> fires(T);
    for (int t = 0; t < T; ++t) fires[t] = (fire_idx[t] ? 1.f : 0.f) + prefix_sum[t] - psf[t];

    // prefix_sum_hidden = cumsum(alphas * concat5) along time
    std::vector<double> psh((size_t)T * H, 0.0);
    for (int h = 0; h < H; ++h) psh[0 * H + h] = (double)alphas_m[0] * concat5[0 * H + h];
    for (int t = 1; t < T; ++t)
        for (int h = 0; h < H; ++h)
            psh[t * H + h] = psh[(t - 1) * H + h] + (double)alphas_m[t] * concat5[t * H + h];

    // Gather frames at fire positions
    std::vector<int> fire_positions;
    fire_positions.reserve(MAX_LABEL_LEN * 2);
    for (int t = 0; t < T; ++t) if (fire_idx[t]) fire_positions.push_back(t);

    int N = (int)fire_positions.size();
    int N_clamped = std::min(N, MAX_LABEL_LEN);
    token_num_out[0] = N_clamped;

    if (N == 0) {
        std::fill(frame_fires_out, frame_fires_out + MAX_LABEL_LEN * H, 0.f);
        return;
    }

    // frames[N, H] = psh at fire positions
    std::vector<float> frames((size_t)N * H);
    for (int k = 0; k < N; ++k) {
        int t = fire_positions[k];
        for (int h = 0; h < H; ++h) frames[k * H + h] = (float)psh[t * H + h];
    }

    // shift_frames = roll(frames, 1) then zero the first slot
    // remain_frames[k, h] = remains[k] * concat5[fire_positions[k], h]
    // where remains = fires - floor(fires), only at fire positions
    std::vector<float> remain(N);
    for (int k = 0; k < N; ++k) {
        int t = fire_positions[k];
        remain[k] = fires[t] - std::floor(fires[t]);
    }
    std::vector<float> remain_frames((size_t)N * H);
    for (int k = 0; k < N; ++k) {
        int t = fire_positions[k];
        for (int h = 0; h < H; ++h) remain_frames[k * H + h] = remain[k] * concat5[t * H + h];
    }

    // Effective frames: frames - shift_frames + shift_remain_frames - remain_frames
    // Since batch_size = 1, shift is trivial (roll(x, 1) shifts by 1 with wraparound; we zero index 0)
    std::vector<float> eff((size_t)N * H, 0.f);
    for (int k = 0; k < N; ++k) {
        for (int h = 0; h < H; ++h) {
            float f_curr = frames[k * H + h];
            float f_prev = (k == 0) ? 0.f : frames[(k - 1) * H + h];
            float r_curr = remain_frames[k * H + h];
            float r_prev = (k == 0) ? 0.f : remain_frames[(k - 1) * H + h];
            eff[k * H + h] = f_curr - f_prev + r_prev - r_curr;
        }
    }

    // Scatter to fixed [1, 100, 512]
    std::fill(frame_fires_out, frame_fires_out + MAX_LABEL_LEN * H, 0.f);
    for (int k = 0; k < N_clamped; ++k)
        for (int h = 0; h < H; ++h)
            frame_fires_out[k * H + h] = eff[k * H + h];
}

// ==== simple char-level Levenshtein (UTF-8 sensitive; assumes tokens are already
// Chinese chars) ====
/**
 * @brief Compute Levenshtein edit distance between token sequences.
 *
 * @param[in] a Reference tokens.
 * @param[in] b Hypothesis tokens.
 * @return Edit distance.
 */
static int lev(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    int n = (int)a.size(), m = (int)b.size();
    if (n == 0) return m;
    if (m == 0) return n;
    std::vector<int> dp(m + 1);
    for (int j = 0; j <= m; ++j) dp[j] = j;
    for (int i = 1; i <= n; ++i) {
        int prev = dp[0]; dp[0] = i;
        for (int j = 1; j <= m; ++j) {
            int cur = dp[j];
            dp[j] = (a[i - 1] == b[j - 1]) ? prev
                                            : 1 + std::min({prev, dp[j], dp[j - 1]});
            prev = cur;
        }
    }
    return dp[m];
}

// Convert utf-8 string to vector of characters (Chinese 3-byte, etc.)
/**
 * @brief Split a UTF-8 string into code-point substrings for CER calculation.
 *
 * @param[in] s UTF-8 text.
 * @return Tokenized UTF-8 code points.
 */
static std::vector<std::string> to_chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        int n = 1;
        if      ((c & 0x80) == 0)    n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

/**
 * @brief Execute manifest-based Paraformer inference and latency evaluation.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Argument values; the optional first value limits utterances.
 * @return Zero on successful evaluation.
 */
int paraformer_main(int argc, char** argv) {
    int N_UTT = (argc > 1) ? std::atoi(argv[1]) : 0;   // 0 = all
    const std::string ROOT = ".";
    const std::string HBM_DIR = ROOT + "/hbm";
    const std::string FEATS_DIR = ROOT + "/feats";

    // Load vocab + manifest
    std::ifstream vf(ROOT + "/tokens.json");
    json vocab_j; vf >> vocab_j;
    std::vector<std::string> vocab = vocab_j.get<std::vector<std::string>>();
    std::ifstream mf(ROOT + "/manifest.json");
    json manifest_j; mf >> manifest_j;
    int total = (int)manifest_j.size();
    if (N_UTT > 0 && N_UTT < total) total = N_UTT;
    std::cout << "vocab: " << vocab.size() << " tokens; manifest: " << total << " utts" << std::endl;

    // Load 3 hbms into one packed handle
    auto t_load = clock_::now();
    std::vector<std::string> files = {
        HBM_DIR + "/encoder_int16.hbm",
        HBM_DIR + "/predictor_int16.hbm",
        HBM_DIR + "/decoder_int16.hbm",
    };
    const char* fp[3] = {files[0].c_str(), files[1].c_str(), files[2].c_str()};
    hbDNNPackedHandle_t packed;
    CHECK(hbDNNInitializeFromFiles(&packed, fp, 3));

    // Get model names
    char const** model_names_c; int32_t model_count;
    CHECK(hbDNNGetModelNameList(&model_names_c, &model_count, packed));
    Model enc, pred, dec;
    for (int i = 0; i < model_count; ++i) {
        std::string mn = model_names_c[i];
        if (mn.find("encoder") != std::string::npos) enc.init(packed, mn);
        else if (mn.find("predictor") != std::string::npos) pred.init(packed, mn);
        else if (mn.find("decoder") != std::string::npos) dec.init(packed, mn);
    }
    std::cout << "loaded 3 hbms in " << ms_since(t_load) << " ms" << std::endl;

    // Locate tensor names
    int enc_in = enc.input_idx("speech");
    int enc_out = enc.output_idx("/encoder/after_norm/Add_1_output_0");
    int pred_in = pred.input_idx("/encoder/after_norm/Add_1_output_0");
    int alphas_out = pred.output_idx("/predictor/Add_output_0");
    int concat5_out = pred.output_idx("/predictor/Concat_5_output_0");
    int dec_enc_in = dec.input_idx("/encoder/after_norm/Add_1_output_0");
    int dec_tn_in = dec.input_idx("token_num");
    int dec_bias_in = dec.input_idx("bias_embed");
    int dec_pre_in = dec.input_idx("onnx::Shape_8609");
    int logits_out = dec.output_idx("logits");

    // BIAS: all zeros [1, 1, 512]
    std::vector<float> bias(512, 0.f);
    dec.write_input(dec_bias_in, bias.data(), bias.size() * sizeof(float));

    // Buffers
    std::vector<float> enc_out_buf(1 * 400 * 512);
    std::vector<float> alphas_buf(401);
    std::vector<float> concat5_buf(1 * 401 * 512);
    std::vector<float> frame_fires(1 * MAX_LABEL_LEN * 512);
    std::vector<int32_t> token_num(1);
    std::vector<float> logits(1 * MAX_LABEL_LEN * 8404);

    int total_c = 0, total_e = 0;
    double avg_enc = 0, avg_pred = 0, avg_cif = 0, avg_dec = 0;
    auto t_start = clock_::now();

    for (int i = 0; i < total; ++i) {
        auto& e = manifest_j[i];
        std::string utt_id = e["utt_id"];
        std::string ref = e["text"];
        int real_T = e["feat_length"];

        // Load feats
        NpyArray sp = load_npy(FEATS_DIR + "/" + utt_id + ".npy");
        if (sp.dtype != "<f4" || sp.num_elements() != 1 * 400 * 560) {
            std::cerr << "bad feat " << utt_id << " dtype=" << sp.dtype << " nelem=" << sp.num_elements() << " shape=";
            for (auto d : sp.shape) std::cerr << d << " ";
            std::cerr << std::endl; continue;
        }

        // Encoder
        enc.write_input(enc_in, sp.data.data(), sp.data.size());
        double t_e = enc.infer();
        enc.read_output(enc_out, enc_out_buf.data(), enc_out_buf.size() * sizeof(float));

        // Predictor (input = encoder output)
        pred.write_input(pred_in, enc_out_buf.data(), enc_out_buf.size() * sizeof(float));
        double t_p = pred.infer();
        pred.read_output(alphas_out, alphas_buf.data(), alphas_buf.size() * sizeof(float));
        pred.read_output(concat5_out, concat5_buf.data(), concat5_buf.size() * sizeof(float));

        // CIF (CPU)
        auto t_c0 = clock_::now();
        cif_numpy(alphas_buf.data(), concat5_buf.data(), real_T,
                  frame_fires.data(), token_num.data());
        double t_c = ms_since(t_c0);

        // Decoder: fill 4 inputs
        dec.write_input(dec_enc_in, enc_out_buf.data(), enc_out_buf.size() * sizeof(float));
        dec.write_input(dec_tn_in, token_num.data(), token_num.size() * sizeof(int32_t));
        dec.write_input(dec_pre_in, frame_fires.data(), frame_fires.size() * sizeof(float));
        double t_d = dec.infer();
        dec.read_output(logits_out, logits.data(), logits.size() * sizeof(float));

        // Argmax + decode. ``@@`` is the FunASR BPE continuation marker and
        // must not be shown in the final transcription.
        int tn = token_num[0];
        std::vector<std::string> hyp_tokens;
        for (int k = 0; k < tn; ++k) {
            const float* row = logits.data() + k * 8404;
            int arg = 0; float best = row[0];
            for (int v = 1; v < 8404; ++v) if (row[v] > best) { best = row[v]; arg = v; }
            const std::string& tok = vocab[arg];
            if (!(tok.size() >= 2 && tok.front() == '<' && tok.back() == '>')) hyp_tokens.push_back(tok);
        }
        std::string hyp_s;
        for (const auto& token : hyp_tokens) {
            std::string merged = token;
            size_t marker = 0;
            while ((marker = merged.find("@@", marker)) != std::string::npos) {
                merged.erase(marker, 2);
            }
            hyp_s += merged;
        }
        auto ref_chars = to_chars(ref);
        auto hyp_chars = to_chars(hyp_s);
        int err = lev(ref_chars, hyp_chars);
        total_c += (int)ref_chars.size();
        total_e += err;
        avg_enc += t_e; avg_pred += t_p; avg_cif += t_c; avg_dec += t_d;

        if ((i + 1) % 20 == 0 || i < 3) {
            std::cout << "[" << (i+1) << "/" << total << "] "
                      << "enc=" << t_e << " pred=" << t_p << " cif=" << t_c << " dec=" << t_d << " ms  ";
            if (total_c > 0) {
                std::cout << "CER=" << (double)total_e / total_c * 100.0 << "%  ";
            } else {
                std::cout << "CER=N/A  ";
            }
            std::cout << "ref='" << ref.substr(0, 30) << "' hyp='" << hyp_s.substr(0, 30) << "'"
                      << std::endl;
        }
    }

    double cer = total_c > 0 ? (double)total_e / total_c * 100.0 : 0.0;
    double wall = ms_since(t_start);
    std::cout << "\n== BOARD (C++ UCP) ==" << std::endl;
    std::cout << "utt=" << total << " chars=" << total_c << " err=" << total_e;
    if (total_c > 0) {
        std::cout << " CER=" << cer << "%";
    } else {
        std::cout << " CER=N/A (no reference text)";
    }
    std::cout << std::endl;
    std::cout << "avg per-stage (ms): enc=" << avg_enc/total << " pred=" << avg_pred/total
              << " cif=" << avg_cif/total << " dec=" << avg_dec/total << std::endl;
    double avg_total = (avg_enc + avg_pred + avg_cif + avg_dec) / total;
    std::cout << "avg per-utt total : " << avg_total << " ms" << std::endl;
    std::cout << "wall-clock elapsed: " << wall/1000.0 << " s (" << wall/total << " ms/utt)" << std::endl;

    // Save results.json
    json out = {
        {"CER", cer},
        {"n_utt", total},
        {"avg_enc_ms", avg_enc/total},
        {"avg_pred_ms", avg_pred/total},
        {"avg_cif_ms", avg_cif/total},
        {"avg_dec_ms", avg_dec/total},
        {"avg_total_ms", avg_total},
    };
    std::ofstream of(ROOT + "/results_board_ucp.json");
    of << out.dump(2);
    std::cout << "saved → results_board_ucp.json" << std::endl;

    enc.release(); pred.release(); dec.release();
    hbDNNRelease(packed);
    return 0;
}
