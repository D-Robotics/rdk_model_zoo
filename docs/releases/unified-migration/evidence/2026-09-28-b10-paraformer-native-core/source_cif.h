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

