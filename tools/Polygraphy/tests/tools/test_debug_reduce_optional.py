# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest


@pytest.mark.parametrize("mode", ["linear", "bisect"])
@pytest.mark.parametrize("op", ["Clip", "Dropout"])
@pytest.mark.parametrize("omit_optional", [False, True])
def test_reduce_omitted_optional_tensors(tmp_path, mode, op, omit_optional):
    tensors = [
        gs.Variable(name, dtype=np.float32, shape=[2]) for name in ("x", "a", "b", "y")
    ]
    x, a, b, y = tensors
    target = gs.Node(op, name="target", inputs=[a], outputs=[b])
    if op == "Clip":
        minimum = (
            gs.Variable.empty()
            if omit_optional
            else gs.Constant("min", np.array(-1, dtype=np.float32))
        )
        target.inputs.extend(
            [minimum, gs.Constant("max", np.array(1, dtype=np.float32))]
        )
    elif omit_optional:
        target.outputs.append(gs.Variable.empty())
    prefix = [
        x,
        gs.Variable("p", dtype=np.float32, shape=[2]),
        gs.Variable("q", dtype=np.float32, shape=[2]),
        a,
    ]
    suffix = [
        b,
        gs.Variable("r", dtype=np.float32, shape=[2]),
        gs.Variable("s", dtype=np.float32, shape=[2]),
        y,
    ]
    graph = gs.Graph(
        nodes=[
            gs.Node("Identity", inputs=[inp], outputs=[out])
            for inp, out in zip(prefix, prefix[1:])
        ]
        + [target]
        + [
            gs.Node("Identity", inputs=[inp], outputs=[out])
            for inp, out in zip(suffix, suffix[1:])
        ],
        inputs=[x],
        outputs=[y],
        opset=13,
        ir_version=10,
    )
    original = gs.export_onnx(graph)
    onnx.checker.check_model(original)
    onnx.save(original, tmp_path / "model.onnx")
    checker = tmp_path / "check.py"
    checker.write_text(
        "import onnx, sys\nfrom pathlib import Path\n"
        "model = onnx.load('polygraphy_debug.onnx')\n"
        "try:\n"
        "    onnx.checker.check_model(model)\n"
        "except Exception as error:\n"
        "    Path('invalid-artifact.txt').write_text(str(error))\n"
        "    raise\n"
        "sys.exit(1 if any(n.name == 'target' for n in model.graph.node) else 0)\n"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "bin" / "polygraphy"),
        "debug",
        "reduce",
        "model.onnx",
        "--mode",
        mode,
        "--output",
        "bad.onnx",
        "--min-good",
        "good.onnx",
        "--check",
        sys.executable,
        str(checker),
    ]
    status = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    (tmp_path / "reduce.log").write_text(status.stdout + status.stderr)
    assert status.returncode == 0, status.stdout + status.stderr
    assert not (tmp_path / "invalid-artifact.txt").exists()
    assert "The following outputs were not found" not in status.stdout
    bad = onnx.load(tmp_path / "bad.onnx")
    good = onnx.load(tmp_path / "good.onnx")
    for model in (bad, good):
        onnx.checker.check_model(model)
        assert all(tensor.name for tensor in (*model.graph.input, *model.graph.output))
    assert [node.op_type for node in bad.graph.node] == [op]
    assert all(node.name != "target" for node in good.graph.node), (
        status.stdout + status.stderr
    )
    assert len(bad.graph.input) == 1
    if op == "Clip" and omit_optional:
        assert bad.graph.node[0].input[1] == ""
    elif op == "Dropout" and omit_optional:
        assert list(bad.graph.node[0].output) == ["b", ""]
