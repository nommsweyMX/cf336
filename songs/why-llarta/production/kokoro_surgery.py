"""Build controllable variants of the Kokoro v1.0 ONNX graph (local graph surgery, no retraining).

The stock export takes (tokens, style, speed) and returns audio only. Kokoro internally predicts
  * a duration (in 25 ms alignment frames = 600 samples @ 24 kHz) for every token,
  * an F0 curve (Hz) and an energy curve N, both at 12.5 ms (300 samples) per point,
and feeds those to its neural decoder. We expose them as outputs and allow overriding them as inputs:
  ana   : outputs audio, dur, f0, n                       (analysis of natural speech)
  dur   : + input dur_in (float32 [T])          -> audio, f0, n  (re-timed, predictor re-run on new alignment)
  sing  : + inputs dur_in, f0_in [1, 2F], n_in [1, 2F] -> audio  (full control: a singing synthesiser)
"""
import os
import onnx
from onnx import helper, TensorProto

DUR = "/encoder/Clip_output_0"         # rounded, clipped durations (float)
F0 = "/encoder/Reshape_output_0"       # predicted F0  [1, 2F]
NN = "/encoder/Reshape_1_output_0"     # predicted N   [1, 2F]


def _rewire(graph, old, new):
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new


def build(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"kokoro-{k}.onnx") for k in ("ana", "dur", "sing")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    for kind, path in paths.items():
        m = onnx.load(src)
        g = m.graph
        if kind == "ana":
            g.output.extend([
                helper.make_tensor_value_info(DUR, TensorProto.FLOAT, None),
                helper.make_tensor_value_info(F0, TensorProto.FLOAT, None),
                helper.make_tensor_value_info(NN, TensorProto.FLOAT, None),
            ])
        else:
            g.input.append(helper.make_tensor_value_info("dur_in", TensorProto.FLOAT, None))
            _rewire(g, DUR, "dur_in")
            if kind == "dur":
                g.output.extend([
                    helper.make_tensor_value_info(F0, TensorProto.FLOAT, None),
                    helper.make_tensor_value_info(NN, TensorProto.FLOAT, None),
                ])
            else:
                g.input.append(helper.make_tensor_value_info("f0_in", TensorProto.FLOAT, None))
                g.input.append(helper.make_tensor_value_info("n_in", TensorProto.FLOAT, None))
                _rewire(g, F0, "f0_in")
                _rewire(g, NN, "n_in")
        onnx.save(m, path)
    return paths


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1], sys.argv[2]))
