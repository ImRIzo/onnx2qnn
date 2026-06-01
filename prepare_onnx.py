#!/usr/bin/env python3
"""
Split YOLOv8 single-output ONNX into boxes + scores outputs.
Auto-detects split tensors and class count from the model graph.

Usage:
  python3 prepare_onnx.py --input best.onnx --output best_ready.onnx
  python3 prepare_onnx.py --input best.onnx --output best_ready.onnx --num-classes 1
"""

import os
# Force Python implementation BEFORE importing onnx
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import onnx
from onnx import helper, TensorProto
import argparse


def prepare(input_path, output_path, num_classes=None):
    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return output_path

    print(f"Loading: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    print(f"  Original outputs: {[o.name for o in graph.output]}")

    # ── Build node lookup by output name ────────────────────────────
    node_by_output = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node

    # ── Auto-detect split tensors from the final Concat node ──────
    concat_node = None
    for node in graph.node:
        if "output0" in node.output and node.op_type == "Concat":
            concat_node = node
            break

    if concat_node is None:
        # Fallback: try finding the final Concat by looking at the output
        for node in graph.node:
            for out_name in node.output:
                for go in graph.output:
                    if out_name == go.name and node.op_type == "Concat":
                        concat_node = node
                        break
                if concat_node:
                    break
            if concat_node:
                break

    if concat_node is None:
        raise RuntimeError(
            "Could not auto-detect split tensors. "
            "Make sure the model has a final Concat node producing the output."
        )

    # ── Trace backward to find the true Mul (boxes) and Sigmoid (scores) ──
    init_names = {init.name for init in graph.initializer}

    def trace_to_source(tensor_name, target_ops):
        """Walk backward through pass-through and arithmetic ops
        to find the semantically meaningful source tensor."""
        current = tensor_name
        passthrough = {'Transpose', 'Reshape', 'Squeeze', 'Unsqueeze',
                       'Flatten', 'Identity'}

        while current in node_by_output:
            node = node_by_output[current]
            if node.op_type in target_ops:
                break  # reached the target operation
            if node.op_type in passthrough:
                current = node.input[0]
            elif node.op_type == 'Mul':
                # Follow the non-constant (data) input
                data_in = next((i for i in node.input if i not in init_names), None)
                current = data_in if data_in else node.input[0]
            else:
                break  # stop at any other op type

        return current

    raw_boxes_name = concat_node.input[0]   # e.g. /model.22/Mul_5_output_0
    raw_scores_name = concat_node.input[1]  # e.g. /model.22/Reshape_...  or Sigmoid

    boxes_name = trace_to_source(raw_boxes_name, {'Mul'})
    scores_name = trace_to_source(raw_scores_name, {'Sigmoid'})

    print(f"  Concat inputs (raw):  boxes={raw_boxes_name}, scores={raw_scores_name}")
    print(f"  Traced to sources:    boxes={boxes_name}, scores={scores_name}")

    # ── Look up value_info for the split tensors ──────────────────
    def find_info(name):
        for vi in list(graph.value_info) + list(graph.output):
            if vi.name == name:
                return vi
        return None

    boxes_vi = find_info(boxes_name)
    scores_vi = find_info(scores_name)

    if boxes_vi is None:
        raise RuntimeError(f"Tensor '{boxes_name}' not found in value_info")
    if scores_vi is None:
        raise RuntimeError(f"Tensor '{scores_name}' not found in value_info")

    # ── Fix INT64 → FLOAT32 (QNN INT8 cannot consume INT64 outputs) ──
    def ensure_float32(tensor_name, vi, role):
        """If the tensor is INT64, insert a Cast → FLOAT32.
        Returns (final_name, final_vi)."""
        if vi.type.tensor_type.elem_type != TensorProto.INT64:
            return tensor_name, vi

        cast_name = tensor_name + "_cast_float"
        print(f"  {role} '{tensor_name}' is INT64 — adding Cast → FLOAT32 as '{cast_name}'")

        cast_node = helper.make_node(
            'Cast',
            inputs=[tensor_name],
            outputs=[cast_name],
            to=TensorProto.FLOAT32,
        )
        graph.node.append(cast_node)

        # Build value_info for the Cast output
        shape_dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        cast_vi = helper.make_tensor_value_info(
            cast_name, TensorProto.FLOAT32, shape_dims,
        )
        for i, src_dim in enumerate(vi.type.tensor_type.shape.dim):
            if src_dim.dim_param:
                cast_vi.type.tensor_type.shape.dim[i].dim_param = src_dim.dim_param

        graph.value_info.append(cast_vi)
        return cast_name, cast_vi

    boxes_name, boxes_vi = ensure_float32(boxes_name, boxes_vi, "boxes")
    scores_name, scores_vi = ensure_float32(scores_name, scores_vi, "scores")

    # ── Determine num_classes from the model if not provided ──────
    auto_detected = False
    if num_classes is None:
        auto_detected = True
        scores_shape = scores_vi.type.tensor_type.shape
        if len(scores_shape.dim) >= 2:
            nc = scores_shape.dim[1].dim_value
            if nc > 0:
                num_classes = nc
        if num_classes is None:
            # Fallback: count from the class score dimension
            num_classes = 80  # safe default for standard YOLOv8n
    label = "auto-detected" if auto_detected else "user-specified"
    print(f"  num_classes: {num_classes} ({label})")

    # ── Replace outputs: remove old, add split outputs ────────────
    graph.output.clear()

    # Preserve the original types and shapes from value_info
    boxes_out = helper.make_tensor_value_info(
        boxes_vi.name,
        boxes_vi.type.tensor_type.elem_type,
        [d.dim_value for d in boxes_vi.type.tensor_type.shape.dim],
    )
    scores_out = helper.make_tensor_value_info(
        scores_vi.name,
        scores_vi.type.tensor_type.elem_type,
        [d.dim_value for d in scores_vi.type.tensor_type.shape.dim],
    )

    # Restore dim_param for symbolic/dynamic dimensions
    for src_vi, out_vi in [
        (boxes_vi, boxes_out),
        (scores_vi, scores_out),
    ]:
        for i, src_dim in enumerate(src_vi.type.tensor_type.shape.dim):
            if src_dim.dim_param:
                out_vi.type.tensor_type.shape.dim[i].dim_param = src_dim.dim_param

    graph.output.extend([boxes_out, scores_out])

    # ── Clean up value_info ───────────────────────────────────────
    # Remove tensors that are now outputs (ONNX forbids overlap)
    # Remove known-problematic INT64 tensors that confuse QNN parsers
    output_names = {boxes_name, scores_name}
    int64_to_remove = {
        '/model.22/Concat_22_output_0',
        '/model.22/Unsqueeze_14_output_0',
    }

    kept = [
        vi for vi in graph.value_info
        if vi.name not in output_names and vi.name not in int64_to_remove
    ]
    graph.value_info.clear()
    graph.value_info.extend(kept)

    onnx.save(model, output_path)

    # Print summary
    b_type_name = TensorProto.DataType.Name(boxes_vi.type.tensor_type.elem_type)
    s_type_name = TensorProto.DataType.Name(scores_vi.type.tensor_type.elem_type)
    b_shape = [d.dim_value or d.dim_param for d in boxes_vi.type.tensor_type.shape.dim]
    s_shape = [d.dim_value or d.dim_param for d in scores_vi.type.tensor_type.shape.dim]
    print(f"  boxes:  {b_type_name} {b_shape}")
    print(f"  scores: {s_type_name} {s_shape}")
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split YOLOv8 ONNX into boxes + scores outputs"
    )
    p.add_argument("--input", required=True, help="Input ONNX model")
    p.add_argument("--output", required=True, help="Output ONNX model")
    p.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of classes (1-80). Auto-detected from model if not specified."
    )
    args = p.parse_args()
    prepare(args.input, args.output, num_classes=args.num_classes)
