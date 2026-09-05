#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import types

import numpy as np
import pytest
import tensorrt as trt

from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import ModelArgs, TrtConfigArgs
from polygraphy.tools.debug.subtool.precision import LinearMarker, Precision


@pytest.fixture()
def network():
    logger = trt.Logger()
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    x = network.add_input("input", trt.float32, (1,))
    constant = network.add_constant((1,), np.ones(1, dtype=np.float32))
    constant.name = "constant_skip"
    constant.get_output(0).name = "bias"
    value = x
    for index in range(3):
        layer = network.add_elementwise(
            value, constant.get_output(0), trt.ElementWiseOperation.SUM
        )
        layer.name = f"add_{index}"
        value = layer.get_output(0)
        value.name = f"value_{index}"
    network.mark_output(value)
    yield network


@pytest.mark.parametrize("mode", ["bisect", "linear"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("threshold", [0, 2, 4, 5])
@pytest.mark.parametrize("replay", [False, True])
def test_reports_successful_layers(network, mode, direction, threshold, replay, capsys):
    tool = Precision()
    config = TrtConfigArgs()
    config.fp16, config.int8, config.tf32 = True, False, False
    model = ModelArgs()
    model.model_type = "onnx"
    tool.arg_groups[TrtConfigArgs] = config
    tool.arg_groups[ModelArgs] = model
    args = types.SimpleNamespace(precision="float32", mode=mode, direction=direction)
    # A preexisting constraint outside the search range must survive reporting.
    network.get_layer(3).precision = trt.float16
    with G_LOGGER.verbosity(G_LOGGER.INFO):
        tool.setup(args, network)
        for _ in range(10):
            if not replay:
                tool.process_network(network)
            success = tool.layer_marker.num_layers_to_mark >= threshold
            if tool.step(success):
                break
        else:
            pytest.fail("Search did not terminate")
    captured = capsys.readouterr()
    output = captured.out
    if threshold > 4:
        assert "Higher-precision layers in the successful configuration" not in output
        return
    assert "Could not find a configuration" not in captured.err
    assert "Higher-precision layers in the successful configuration" in output
    report = output.split("Higher-precision layers in the successful configuration", 1)[
        1
    ]
    # Bisect starts with all layers and does not probe zero; its lower bound is one.
    count = max(1, threshold) if mode == "bisect" else threshold
    indices = range(count) if direction == "forward" else range(4 - count, 4)
    expected = set(indices) - {0}
    assert "constant_skip" not in report
    for index in range(1, 4):
        assert (f"add_{index - 1}" in report) == (index in expected)
        if index in expected:
            assert f"value_{index - 1}" in report
            assert network.get_layer(index).precision == trt.float32
    if 3 not in expected:
        assert network.get_layer(3).precision == trt.float16


def test_report_excludes_shape_and_boolean_outputs(network, capsys):
    x = network.get_input(0)
    shape = network.add_shape(x)
    shape.name = "shape_skip"
    boolean = network.add_elementwise(x, x, trt.ElementWiseOperation.EQUAL)
    boolean.name = "boolean_skip"
    network.mark_output(shape.get_output(0))
    network.mark_output(boolean.get_output(0))
    original_shape_precision = shape.precision
    original_boolean_precision = boolean.precision
    tool = Precision()
    tool.network = network
    tool.original_precisions = {}
    tool.precision = trt.float32
    tool.layer_marker = LinearMarker(len(network), "forward")
    tool.layer_marker.num_layers_to_mark = len(network)
    with G_LOGGER.verbosity(G_LOGGER.INFO):
        assert tool.step(True)
    report = capsys.readouterr().out
    assert "(3 marked, DataType.FLOAT)" in report
    assert "shape_skip" not in report
    assert "boolean_skip" not in report
    assert shape.precision == original_shape_precision
    assert boolean.precision == original_boolean_precision
