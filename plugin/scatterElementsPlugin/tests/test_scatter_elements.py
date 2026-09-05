# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run against SCATTER_PLUGIN_LIBRARY with TensorRT matching the library build."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "samples/python"))
import common_runtime as common


@pytest.fixture(scope="module")
def creator():
    library = ctypes.CDLL(os.environ["SCATTER_PLUGIN_LIBRARY"])
    library.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.initLibNvInferPlugins.restype = ctypes.c_bool
    assert library.initLibNvInferPlugins(None, b"scatter_test")
    creator = trt.get_plugin_registry().get_creator(
        "ScatterElements", "2", "scatter_test"
    )
    assert creator is not None
    yield creator


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.int64])
@pytest.mark.parametrize("reduction", ["add", "mul", "min", "max"])
@pytest.mark.parametrize(
    "axis",
    [1, -1, 0, -2],
    ids=["positive-axis", "negative-axis", "first-axis", "negative-first-axis"],
)
@pytest.mark.parametrize("negative_indices", [False, True])
def test_scatter_negative_indices(creator, dtype, reduction, axis, negative_indices):
    logger = trt.Logger(trt.Logger.ERROR)
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "axis", np.array([axis], dtype=np.int32), trt.PluginFieldType.INT32
            ),
            trt.PluginField("reduction", reduction.encode(), trt.PluginFieldType.CHAR),
        ]
    )
    plugin = creator.create_plugin("scatter", fields, trt.TensorRTPhase.BUILD)
    assert plugin is not None
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    data = network.add_input(
        "data",
        {
            np.float32: trt.float32,
            np.float16: trt.float16,
            np.int32: trt.int32,
            np.int64: trt.int64,
        }[dtype],
        (3, 3),
    )
    update_shape = (3, 2) if axis % 2 == 1 else (2, 3)
    indices = network.add_input("indices", trt.int64, update_shape)
    updates = network.add_input("updates", data.dtype, update_shape)
    layer = network.add_plugin_v3([data, indices, updates], [], plugin)
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    inputs, outputs, bindings = common.allocate_buffers(engine)
    names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    ]
    operation = {
        "add": np.add,
        "mul": np.multiply,
        "min": np.minimum,
        "max": np.maximum,
    }[reduction]
    try:
        with common.CudaStreamContext() as stream:
            # For axis=1, negative indices occur after the first row, so the
            # baseline reproducer stays within the allocation. Run other axes
            # only with the fixed library.
            index_values = (
                [[0, 0], [-3, -1], [-1, -1]]
                if negative_indices
                else [[0, 0], [0, 2], [2, 2]]
            )
            data_values = np.arange(1, 10, dtype=dtype).reshape(3, 3)
            update_values = np.array([[2, 3], [4, 2], [1, 2]], dtype=dtype)
            index_values = np.array(index_values, dtype=np.int64)
            expected = data_values.copy()
            if axis % 2 == 0:
                index_values = np.ascontiguousarray(index_values.T)
                update_values = np.ascontiguousarray(update_values.T)
                coordinates = (index_values, np.arange(3)[None, :])
            else:
                coordinates = (np.arange(3)[:, None], index_values)
            operation.at(expected, coordinates, update_values)
            feeds = {
                "data": data_values,
                "indices": index_values,
                "updates": update_values,
            }
            for name, mem in zip(names, inputs):
                mem.host = feeds[name].ravel()
            actual = common.do_inference(
                context, engine, bindings, inputs, outputs, stream
            )[0].reshape(3, 3)
            np.testing.assert_array_equal(actual, expected)
    finally:
        common.free_buffers(inputs, outputs)
