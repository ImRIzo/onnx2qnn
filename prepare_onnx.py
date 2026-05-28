#!/usr/bin/env python3
"""
Split YOLOv8 single-output ONNX into boxes + scores outputs.
No QAI Hub dependency — pure onnx library.

Usage:
  python3 prepare_onnx.py --input best.onnx --output best_ready.onnx
"""

import os
# Force Python implementation BEFORE importing onnx
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import onnx
from onnx import helper, TensorProto
import argparse

def prepare(input_path, output_path):
    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return output_path
    print(f"Loading: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    # Now .clear() will work with Python backend
    print(f"  Original outputs: {[o.name for o in graph.output]}")
    graph.output.clear()

    # Expose intermediate tensors
    boxes_name  = '/model.22/Mul_2_output_0'
    scores_name = '/model.22/Sigmoid_output_0'

    boxes_out = helper.make_tensor_value_info(boxes_name, TensorProto.FLOAT, [1, 4, 8400])
    scores_out = helper.make_tensor_value_info(scores_name, TensorProto.FLOAT, [1, 80, 8400])
    graph.output.extend([boxes_out, scores_out])

    # Clear value_info (Ultralytics duplicates outputs here — QNN parser rejects it)
    graph.value_info.clear()

    onnx.save(model, output_path)
    print(f"Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    prepare(args.input, args.output)
