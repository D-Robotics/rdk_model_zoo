"""
Step 11: End-to-end CER evaluation on a labeled test set.

Runs one of the two pipelines and computes CER against reference transcripts:
  - FP32:  plain onnx (encoder_only, predictor_only, decoder_only_final)
  - INT16: HMCT-quantized ptq_model.onnx variants (via `hmct.executor.ORTExecutor`)

Usage:
    # FP32 pipeline
    python 11_eval_pipeline.py --pipeline fp32 --eval_dir ./aishell_eval

    # INT16 pipeline (must be run inside the toolchain docker with HMCT installed)
    python 11_eval_pipeline.py --pipeline int16 --eval_dir ./aishell_eval

Expected file layout:
    ./aishell_eval/
    ├── feats/*.npy          # each [1, 400, 560]
    └── manifest.json        # list of {utt_id, text, feat_length}

Prints running CER; final CER = total_edit_distance / total_reference_chars.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cif_numpy import cif_numpy


def load_sessions(pipeline, model_paths):
    if pipeline == "fp32":
        import onnxruntime as ort
        enc = ort.InferenceSession(model_paths["enc"], providers=["CPUExecutionProvider"])
        pred = ort.InferenceSession(model_paths["pred"], providers=["CPUExecutionProvider"])
        dec = ort.InferenceSession(model_paths["dec"], providers=["CPUExecutionProvider"])
        def run(sess, feed):
            out = sess.run(None, feed)
            return dict(zip([o.name for o in sess.get_outputs()], out))
    else:
        from hmct.executor import ORTExecutor as ORT
        enc = ORT(model_paths["enc"]).create_session()
        pred = ORT(model_paths["pred"]).create_session()
        dec = ORT(model_paths["dec"]).create_session()
        def run(sess, feed):
            return sess.forward(feed)
    return enc, pred, dec, run


def decode_tokens(vocab, ids):
    out = []
    for tid in ids:
        t = vocab[int(tid)]
        if t.startswith("<") and t.endswith(">"):
            continue
        out.append(t)
    return "".join(out)


def levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[-1]


def main(args):
    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    with open(f"{args.eval_dir}/manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    paths = {"enc": args.enc, "pred": args.pred, "dec": args.dec}
    enc, pred, dec, run = load_sessions(args.pipeline, paths)
    BIAS = np.zeros((1, 1, 512), dtype=np.float32)

    total_c = total_e = 0
    results = []
    t0 = time.time()
    for i, e in enumerate(manifest):
        feat = f"{args.eval_dir}/feats/{e['utt_id']}.npy"
        if not os.path.exists(feat):
            continue
        speech = np.load(feat)
        real_T = e.get("feat_length", None)

        enc_out = run(enc, {"speech": speech})["/encoder/after_norm/Add_1_output_0"]
        pred_r = run(pred, {"/encoder/after_norm/Add_1_output_0": enc_out})
        alphas = pred_r["/predictor/Add_output_0"]
        concat5 = pred_r["/predictor/Concat_5_output_0"]

        frame_fires, token_num = cif_numpy(alphas, concat5, real_T=real_T)

        dec_r = run(dec, {
            "/encoder/after_norm/Add_1_output_0": enc_out,
            "token_num": token_num,
            "bias_embed": BIAS,
            "onnx::Shape_8609": frame_fires,
        })
        logits = dec_r["logits"]
        tn = int(token_num[0])
        ids = np.argmax(logits[0, :tn], axis=-1)
        hyp = decode_tokens(vocab, ids)
        ref = e["text"]
        err = levenshtein(list(ref), list(hyp))
        total_e += err; total_c += len(ref)
        results.append({"utt_id": e["utt_id"], "ref": ref, "hyp": hyp, "err": err, "n_ref": len(ref)})
        if (i+1) % 20 == 0 or i < 5:
            cer_run = total_e / max(1, total_c) * 100
            print(f"[{i+1}/{len(manifest)}] CER={cer_run:.2f}%  ref='{ref[:15]}'  hyp='{hyp[:25]}'", flush=True)

    cer = total_e / max(1, total_c) * 100
    print(f"\n== {args.pipeline.upper()} FINAL ==  utt={len(results)}  chars={total_c}  err={total_e}  CER={cer:.3f}%  t={time.time()-t0:.1f}s")

    out_json = f"{args.eval_dir}/results_{args.pipeline}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"CER": cer, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"saved → {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["fp32", "int16"], required=True)
    p.add_argument("--eval_dir", default="./aishell_eval")
    p.add_argument("--vocab",    default="./models/paraformer/tokens.json")
    p.add_argument("--enc",      help="encoder onnx (FP32) or ptq_model.onnx (INT16)")
    p.add_argument("--pred",     help="predictor onnx / ptq_model.onnx")
    p.add_argument("--dec",      help="decoder onnx / ptq_model.onnx")
    args = p.parse_args()
    if not args.enc:
        args.enc = "./out/encoder_only.onnx" if args.pipeline == "fp32" else "./encoder_int16_output/paraformer_encoder_int16_ptq_model.onnx"
    if not args.pred:
        args.pred = "./out/predictor_only.onnx" if args.pipeline == "fp32" else "./predictor_int16_output/predictor_int16_ptq_model.onnx"
    if not args.dec:
        args.dec = "./out/decoder_only_final.onnx" if args.pipeline == "fp32" else "./decoder_int16_output/decoder_int16_ptq_model.onnx"
    main(args)
