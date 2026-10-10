# MobileNet C++ benchmark

This native benchmark loads a model once per pipeline stream and measures
batch-one inference on X5 (libdnn) or S100/S100P/S600 (libhbucp). The artifact must expose
square NV12 video-range inputs and one contiguous 1000-class float output.
The input side (for example 224 or 256) is read from the model; input tensor
shapes, strides, data types and memory sizes are checked before use.

Build on the board with its installed matching SDK, CMake, g++ and OpenCV:

```bash
cmake -S utils/tools/mobilenet/cpp -B /path/to/new-build \
  -DCMAKE_BUILD_TYPE=Release -DTARGET_S100=ON
cmake --build /path/to/new-build -j2
/path/to/new-build/mobilenet_benchmark \
  /path/to/model.hbm /path/to/images.txt /path/to/new-output/run \
  2 200 20 6
```

`-DTARGET_S100=ON` selects the S-series UCP path used on S100, S100P and S600.
On S600 (nash-p) also pass `-DNV12_ROW_ALIGN=64`: its runtime requires NV12 rows
aligned to 64 bytes, so 224-byte rows are copied into 256-byte padded rows.
S100/S100P keep the default 32, which leaves rows packed. Use
`-DTARGET_S100=OFF` and a `.bin` for X5. The positional arguments are model,
image list (one path per line), output prefix, streams (1 or 2), measured frames
**per stream**, warmup frames per stream, OpenCV CPU threads, and an optional shorter-edge
resize size (`int(size / crop_pct)`: 256 for Small, the default, 235 for
Medium-224 and 269 for Large-256). Output prefix
parents must exist. Use a fresh output prefix for every run. Use all online CPU
threads for the maximum-performance condition and record CPU/BPU frequency,
governor and SDK version separately; this executable does not change them.

Images are decoded before timing. Each timed frame performs Pillow-compatible
antialiased bicubic shorter-edge resize, center crop to the model input size, OpenCV I420
to NV12 packing, input upload/cache clean, model execution/wait, output cache
invalidation and Top-5 selection. Preprocess includes input upload, runtime
includes submit/wait/task release, and postprocess includes output readback and
Top-5. Loading, decoding, drawing, output files and warmup are excluded.

Streams have independent models, tensors and workers. Measured work begins
behind one barrier; throughput uses all completed frames divided by their
common wall time. OpenCV's process-wide pool is shared by both workers. CSV
contains every stage time; JSON contains frame counts and wall-clock throughput.
The first cycle also saves logits and crop PNGs for Python/C++ equivalence.
The three images used in P2 are lossless copies of already decoded evaluation
images so JPEG decoder differences do not obscure geometry or Runtime parity.

This is an accuracy-preserving reference pipeline. Its scalar fixed-point
bicubic kernel is not claimed to be the fastest available implementation.
The 0.372011136 GFLOPs figure describes Conv/Gemm multiply-adds in the source
model (two FLOPs per MAC), excluding this CPU preprocessing.
