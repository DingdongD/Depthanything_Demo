#!/usr/bin/env python3
"""Remove redundant split-QKV -> Concat -> three Slice round trips.

After fused QKV projections are split into independent Q/K/V linears, rebuilding
the 1152-wide tensor only to slice it immediately is unnecessary and is unsafe
on U250.  This pass connects each branch output directly to its original
reshape path and dead-code-eliminates the dynamic slicing scaffold.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import onnx


def bypass(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    rewires: dict[str, str] = {}
    count = 0
    for concat in nodes:
        if concat.op_type != "Concat" or not concat.name.endswith("/QKVSplit"):
            continue
        if len(concat.input) != 3 or len(concat.output) != 1:
            raise ValueError(f"{concat.name}: expected three inputs and one output")
        slices = [
            node
            for node in consumers.get(concat.output[0], [])
            if node.op_type == "Slice"
        ]
        slices.sort(key=nodes.index)
        if len(slices) != 3:
            raise ValueError(
                f"{concat.name}: expected three direct Q/K/V Slice users, got "
                f"{[node.name for node in slices]}"
            )
        for branch_output, slice_node in zip(concat.input, slices):
            rewires[slice_node.output[0]] = branch_output
        count += 1

    if not count:
        raise ValueError("no split-QKV reconstruction Concat nodes found")
    for node in nodes:
        for index, input_name in enumerate(node.input):
            if input_name in rewires:
                node.input[index] = rewires[input_name]

    # Backward DCE also removes Shape/Gather/arithmetic used only to construct
    # the now-deleted Slice endpoints.
    producer = {output: node for node in nodes for output in node.output}
    initializers = {value.name for value in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    required: set[int] = set()
    pending = [value.name for value in model.graph.output]
    while pending:
        tensor = pending.pop()
        if tensor in graph_inputs or tensor in initializers:
            continue
        node = producer.get(tensor)
        if node is None:
            raise ValueError(f"missing producer for live tensor {tensor!r}")
        identifier = id(node)
        if identifier in required:
            continue
        required.add(identifier)
        pending.extend(name for name in node.input if name)

    del model.graph.node[:]
    model.graph.node.extend(copy.deepcopy(node) for node in nodes if id(node) in required)
    used_initializers = {name for node in model.graph.node for name in node.input}
    kept = [value for value in model.graph.initializer if value.name in used_initializers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.input), load_external_data=True)
    count = bypass(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(str(args.output) + ".data")
    if args.output.exists():
        args.output.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
    )
    print(f"bypassed_qkv_reconcat={count}")
    print(args.output)


if __name__ == "__main__":
    main()
