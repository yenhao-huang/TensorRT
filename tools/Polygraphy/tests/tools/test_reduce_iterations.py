# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from polygraphy.json import save_json


@pytest.fixture()
def model(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "y"], ["sum"]),
            helper.make_node("Identity", ["sum"], ["out"]),
        ],
        "reduce",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in ["x", "y"]],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
        ),
        path,
    )
    return path


def run_reduce(model, tmp_path, options):
    checker = tmp_path / "check.py"
    checker.write_text(
        "from pathlib import Path\nPath('checked').touch()\nraise SystemExit(1)\n"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from polygraphy.tools import main; raise SystemExit(main())",
            "debug",
            "reduce",
            str(model),
            "-o",
            str(tmp_path / "reduced.onnx"),
            *options,
            "--check",
            sys.executable,
            str(checker),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("mode", ["linear", "bisect"])
@pytest.mark.parametrize(
    "kind", ["list", "generator", "iterator", "bounded", "json", "random"]
)
def test_reject_multiple_iterations(model, tmp_path, mode, kind):
    if kind == "random":
        options = ["--iterations", "2"]
    elif kind == "json":
        path = tmp_path / "inputs.json"
        save_json(
            [
                {n: np.array([i], dtype=np.float32) for n in ["x", "y"]}
                for i in range(2)
            ],
            str(path),
        )
        options = ["--load-inputs", str(path)]
    else:
        script = tmp_path / "loader.py"
        prefix = 'import numpy as np\ndef sample(i):\n    return {n: np.array([i], dtype=np.float32) for n in ["x", "y"]}\n'
        body = {
            "list": "def load_data():\n    return [sample(1), sample(2)]\n",
            "generator": "def load_data():\n    yield sample(1)\n    yield sample(2)\n",
            "iterator": "data = iter([sample(1), sample(2)])\ndef load_data():\n    return data\n",
            "bounded": "def load_data():\n    yield sample(1)\n    yield sample(2)\n    raise AssertionError('Read beyond second sample')\n",
        }[kind]
        script.write_text(prefix + body)
        options = ["--data-loader-script", str(script)]
    result = run_reduce(model, tmp_path, ["--mode", mode, *options])
    assert result.returncode != 0
    assert "only supports a single input iteration" in result.stdout + result.stderr
    assert "--no-reduce-inputs" in result.stdout + result.stderr
    assert not (tmp_path / "checked").exists()
    assert not (tmp_path / "reduced.onnx").exists()


@pytest.mark.parametrize("mode", ["linear", "bisect"])
def test_one_shot_sample_reused_for_fallback(model, tmp_path, mode):
    script = tmp_path / "loader.py"
    script.write_text("""import numpy as np
calls = 0
data = iter([{n: np.array([17], dtype=np.float32) for n in ["x", "y"]}])
def load_data():
    global calls
    calls += 1
    assert calls == 1, "Loader factory called again"
    return data
""")
    result = run_reduce(
        model,
        tmp_path,
        [
            "--mode",
            mode,
            "--data-loader-script",
            str(script),
            "--force-fallback-shape-inference",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "checked").exists()
    assert (tmp_path / "reduced.onnx").exists()
    assert "Running fallback shape inference" in result.stdout


@pytest.mark.parametrize("mode", ["linear", "bisect"])
@pytest.mark.parametrize("kind", ["random", "generator", "json"])
def test_allow_multiple_without_input_reduction(model, tmp_path, mode, kind):
    if kind == "random":
        data_options = ["--iterations", "2"]
    elif kind == "generator":
        script = tmp_path / "loader.py"
        script.write_text(
            'import numpy as np\ndef load_data():\n    for i in range(2):\n        yield {n: np.array([i], dtype=np.float32) for n in ["x", "y"]}\n'
        )
        data_options = ["--data-loader-script", str(script)]
    else:
        path = tmp_path / "inputs.json"
        save_json(
            [
                {n: np.array([i], dtype=np.float32) for n in ["x", "y"]}
                for i in range(2)
            ],
            str(path),
        )
        data_options = ["--load-inputs", str(path)]
    result = run_reduce(
        model,
        tmp_path,
        [
            "--mode",
            mode,
            *data_options,
            "--no-reduce-inputs",
            "--force-fallback-shape-inference",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reduced = onnx.load(tmp_path / "reduced.onnx")
    assert [value.name for value in reduced.graph.input] == ["x", "y"]


def test_empty_loader_error(model, tmp_path):
    script = tmp_path / "loader.py"
    script.write_text("def load_data():\n    return iter([])\n")
    result = run_reduce(model, tmp_path, ["--data-loader-script", str(script)])
    assert result.returncode != 0
    assert "data loader is empty" in result.stdout + result.stderr
    assert not (tmp_path / "checked").exists()
