#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import tensorrt as trt
from cuda.bindings import runtime as cudart

spec = importlib.util.spec_from_file_location("onnx_helper", Path(__file__).parents[1] / "onnx_helper.py")
onnx_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onnx_helper)


def cuda_call(result):
    error, *values = result
    assert error == cudart.cudaError_t.cudaSuccess
    return values[0] if values else None


def save_model(path, dtype=np.float32, dynamic=False, op="MatMul"):
    weights = np.array([[1, 2], [-1, 0.5]], dtype=dtype)
    tensor_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    shape = [None if dynamic else 1, 2]
    graph = helper.make_graph(
        [helper.make_node(op, ["X", "weights"], ["Y"])],
        "test",
        [helper.make_tensor_value_info("X", tensor_type, shape)],
        [helper.make_tensor_value_info("Y", tensor_type, shape)],
        [numpy_helper.from_array(weights, name="weights")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), path)
    return weights


@pytest.mark.parametrize("dtype,convert", [(np.float32, True), (np.float32, False), (np.float16, False)])
def test_build_and_infer(tmp_path, dtype, convert):
    model_path, engine_path = tmp_path / "model.onnx", tmp_path / "model.engine"
    weights = save_model(model_path, dtype)
    serialized, logger = onnx_helper.convert_onnx_to_engine(model_path, engine_path, fp16_mode=convert)
    assert engine_path.read_bytes() == bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    x = np.array([[2, 4]], dtype=dtype)
    y = np.empty_like(x)
    assert np.dtype(trt.nptype(engine.get_tensor_dtype("X"))) == x.dtype
    assert np.dtype(trt.nptype(engine.get_tensor_dtype("Y"))) == y.dtype
    device_in = device_out = stream = None
    try:
        device_in = cuda_call(cudart.cudaMalloc(x.nbytes))
        device_out = cuda_call(cudart.cudaMalloc(y.nbytes))
        stream = cuda_call(cudart.cudaStreamCreate())
        assert context.set_tensor_address("X", device_in)
        assert context.set_tensor_address("Y", device_out)
        cuda_call(
            cudart.cudaMemcpyAsync(
                device_in, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
            )
        )
        assert context.execute_async_v3(stream)
        cuda_call(
            cudart.cudaMemcpyAsync(
                y.ctypes.data, device_out, y.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
            )
        )
        cuda_call(cudart.cudaStreamSynchronize(stream))
        np.testing.assert_array_equal(y, x @ weights)
    finally:
        if stream is not None:
            cuda_call(cudart.cudaStreamSynchronize(stream))
            cuda_call(cudart.cudaStreamDestroy(stream))
        for device in [device_in, device_out]:
            if device is not None:
                cuda_call(cudart.cudaFree(device))


@pytest.mark.parametrize(
    "dynamic,op,message", [(False, "NoSuchOperator", "Failed to parse"), (True, "MatMul", "Failed to build")]
)
def test_failed_build_does_not_write_engine(tmp_path, dynamic, op, message):
    model_path, engine_path = tmp_path / "model.onnx", tmp_path / "model.engine"
    save_model(model_path, dynamic=dynamic, op=op)
    with pytest.raises(RuntimeError, match=message):
        onnx_helper.convert_onnx_to_engine(model_path, engine_path, fp16_mode=False)
    assert not engine_path.exists()
