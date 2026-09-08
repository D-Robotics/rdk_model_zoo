"""Evaluate fixed-shape HBM on the board through the SDK's local RPC service."""
import argparse
import json
import math
import time
from pathlib import Path

import grpc
import numpy as np
from rpc_protocol import frame_pb2 as pb


DTYPES = {"S8": np.int8, "S16": np.int16, "S32": np.int32,
          "S64": np.int64, "F16": np.float16, "F32": np.float32}


def dtype(info):
    """Return the NumPy storage dtype declared by HBM metadata."""
    return DTYPES[pb.DataType.Name(info.data_type).removeprefix("DATA_TYPE_")]


def scales(info):
    """Reshape quantization parameters for broadcasting over a tensor."""
    q = info.quanti_scale_info
    shape = [1] * len(info.valid_shape)
    shape[q.quantizeAxis] = len(q.scale)
    return (np.asarray(q.scale, dtype=np.float32).reshape(shape),
            np.asarray(q.zero_point, dtype=np.float32).reshape(shape))


def quantize(data, info):
    """Match the SDK input quantization using round-to-nearest-even."""
    if info.quanti_scale_info.HasField("quantizeAxis"):
        scale, zero = scales(info)
        limits = np.iinfo(dtype(info))
        data = np.clip(np.rint(data / scale) + zero, limits.min, limits.max)
    return np.asarray(data, dtype=dtype(info))


def unpack(tensor, info):
    """Decode one RPC output and apply its quantization scale."""
    data = np.frombuffer(tensor.data, dtype=dtype(info)).reshape(tensor.properties.valid_shape)
    assert tuple(data.shape) == tuple(info.valid_shape)
    if info.quanti_scale_info.HasField("quantizeAxis"):
        scale, zero = scales(info)
        data = (data.astype(np.float32) - zero) * scale
    return data


def nll(logits, labels):
    """Sum next-token cross entropy in float64 after float32 log-softmax."""
    scores = logits.reshape(-1, logits.shape[-1])[:len(labels)].astype(np.float32)
    shifted = scores - scores.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    return losses.astype(np.float64).sum().item()


def main():
    """Evaluate all requested test segments with a local SDK RPC server."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=40279, help='Local SDK RPC server port')
    parser.add_argument('--hbm', type=Path, required=True, help='S600 HBM file path')
    parser.add_argument('--embedding', type=Path, required=True, help='FP16 embedding file path')
    parser.add_argument('--samples', type=int, default=140, help='Number of segments; only 140 is a full evaluation')
    parser.add_argument('--output', type=Path, default=Path('board-local-ppl.json'), help='Per-segment and final JSON result')
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help='Prepared input_ids.npy and masks.npy directory')
    args = parser.parse_args()
    assert 0 < args.samples <= 140
    channel = grpc.insecure_channel(f'127.0.0.1:{args.port}', options=[
        ('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    communicate = channel.unary_unary('/RPC.GrpcCommu/Communicate',
        request_serializer=pb.Frame.SerializeToString, response_deserializer=pb.Frame.FromString)
    frame = pb.Frame(frame_id=0, message_type=pb.MODEL_LOAD)
    frame.load_info.model_file.append(str(args.hbm))
    loaded = communicate(frame, timeout=200)
    assert not loaded.status, loaded.status
    model = next(m for m in loaded.load_info.model_properties if m.model_name == 'prefill')
    assert len(model.inputs) == 87 and len(model.outputs) == 85
    print('LOCAL_MODEL_LOADED', model.model_name, flush=True)
    ids = np.load(args.data_dir/'input_ids.npy')
    masks = np.load(args.data_dir/'masks.npy')
    embedding = np.memmap(args.embedding, mode='r', dtype=np.float16, shape=(130560, 2048))
    assert ids.size // 2048 == 140
    sample_results = []
    frame_id = 1
    started = time.monotonic()
    for index in range(args.samples):
        tokens = ids[:, index*2048:(index+1)*2048]
        caches = [np.zeros(tuple(p.valid_shape), dtype=np.float32) for p in model.inputs[3:]]
        loss = 0.0
        for ci, start in enumerate(range(0, 2048, 256)):
            arrays = [embedding[tokens[:, start:start+256]],
                      np.arange(start, start+256).reshape(1, 256), masks[ci]] + caches
            frame = pb.Frame(frame_id=frame_id, time_stamp=int(time.time()), message_type=pb.MODEL_INFERENCE)
            frame_id += 1
            frame.infer_info.model_name = 'prefill'
            for data, prop in zip(arrays, model.inputs, strict=True):
                converted = quantize(data, prop)
                assert tuple(converted.shape) == tuple(prop.valid_shape), (prop.name, converted.shape, prop.valid_shape)
                frame.infer_info.inputs.add(data=converted.tobytes())
            for prop in model.outputs:
                frame.infer_info.outputs.add()
            response = communicate(frame, timeout=200)
            assert not response.status, response.status
            outputs = [unpack(t, p) for t, p in zip(response.infer_info.outputs, model.outputs, strict=True)]
            assert outputs[0].shape == (1, 256, 130560)
            count = min(256, 2048-start-1)
            loss += nll(outputs[0], tokens[0, start+1:start+1+count])
            caches = [np.concatenate([old[:, 256:], new], axis=1)
                      for old, new in zip(caches, outputs[1:], strict=True)]
            del response, outputs, frame, arrays
        assert math.isfinite(loss)
        sample_results.append({'index': index, 'nll': loss, 'predicted_tokens': 2047})
        total = sum(s['nll'] for s in sample_results)
        result = {'perplexity': math.exp(total/(len(sample_results)*2047)),
                  'num_samples': len(sample_results), 'seq_len': 2048, 'chunk_size': 256,
                  'total_nll': total, 'predicted_tokens': len(sample_results)*2047,
                  'elapsed_seconds': time.monotonic()-started, 'samples': sample_results}
        args.output.write_text(json.dumps(result, indent=2))
        print(f"LOCAL_PPL_SAMPLE {index+1}/{args.samples} nll={loss:.6f} running_ppl={result['perplexity']:.6f} elapsed={result['elapsed_seconds']:.2f}", flush=True)
    print('LOCAL_PPL_COMPLETE', args.output, flush=True)
    release = pb.Frame(frame_id=frame_id, message_type=pb.MODEL_RELEASE)
    release.load_info.model_file.append(str(args.hbm))
    released = communicate(release, timeout=200)
    assert not released.status, released.status
    channel.close()


if __name__ == '__main__':
    main()
