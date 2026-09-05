# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real GPU regression tests; RESIZE_WITH_PAD_LIBRARY points to the built test library."""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "samples" / "python"))
from common_runtime import DeviceMem, memcpy_host_to_device, memcpy_device_to_host


@pytest.fixture(scope="module")
def creator():
    library = ctypes.CDLL(os.environ["RESIZE_WITH_PAD_LIBRARY"])
    if hasattr(library, "initLibNvInferPlugins"):
        initialize = library.initLibNvInferPlugins
        initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        initialize.restype = ctypes.c_bool
        assert initialize(None, b"")
    creator = trt.get_plugin_registry().get_creator("ResizeWithPadPlugin", "1", "")
    assert creator is not None
    yield creator
    del library


def make_plugin(creator, **options):
    values = {"output_h": 9, "output_w": 11, **options}
    arrays = {
        k: np.asarray(v, dtype=np.float32 if k == "pad_value" else np.int32).reshape(-1)
        for k, v in values.items()
    }
    fields = [
        trt.PluginField(
            k,
            v,
            (
                trt.PluginFieldType.FLOAT32
                if k == "pad_value"
                else trt.PluginFieldType.INT32
            ),
        )
        for k, v in arrays.items()
    ]
    return creator.create_plugin(
        "letterbox", trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD
    )


def reference(
    x,
    layout,
    interpolation,
    pad_mode,
    swap_rb,
    normalize,
    pad_value=114.0,
    output=(9, 11),
):
    """CPU image-domain reference: resize first, then place into a filled canvas."""
    x = x.transpose(0, 3, 1, 2) if layout else x
    n, c, h, w = x.shape
    oh, ow = output
    scale = min(oh / h, ow / w)
    rh, rw = min(oh, max(1, int(np.floor(h * scale + 0.5)))), min(
        ow, max(1, int(np.floor(w * scale + 0.5)))
    )
    top, left = (0, 0) if pad_mode else ((oh - rh) // 2, (ow - rw) // 2)
    result = np.empty((n, c, oh, ow), dtype=np.float64)
    result[:] = np.broadcast_to(np.asarray(pad_value), (c,))[None, :, None, None]
    source = x[:, ::-1] if swap_rb else x
    for y in range(rh):
        for xx in range(rw):
            if interpolation == 0:
                value = source[:, :, y * h // rh, xx * w // rw]
            else:
                sy, sx = max(0.0, (y + 0.5) * h / rh - 0.5), max(
                    0.0, (xx + 0.5) * w / rw - 0.5
                )
                y0, x0 = min(h - 1, int(sy)), min(w - 1, int(sx))
                y1, x1 = min(h - 1, y0 + 1), min(w - 1, x0 + 1)
                dy, dx = sy - y0, sx - x0
                a, b = source[:, :, y0, x0].astype(float), source[:, :, y0, x1].astype(
                    float
                )
                d, e = source[:, :, y1, x0].astype(float), source[:, :, y1, x1].astype(
                    float
                )
                value = (
                    a * (1 - dx) * (1 - dy)
                    + b * dx * (1 - dy)
                    + d * (1 - dx) * dy
                    + e * dx * dy
                )
            result[:, :, y + top, xx + left] = value
    if normalize:
        result /= 255.0
    return result, np.tile([scale, top, left, 0.0], (n, 1))


def build(creator, dtype, layout, channels=3, expect_failure=False, **options):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    shape = (-1, -1, -1, channels) if layout else (-1, channels, -1, -1)
    tensor = network.add_input("image", dtype, shape)
    # TensorRT permits UINT8 only at network boundaries; convert on GPU.
    plugin_input = (
        network.add_cast(tensor, trt.float32).get_output(0)
        if dtype == trt.uint8
        else tensor
    )
    plugin = make_plugin(creator, layout=layout, **options)
    assert plugin is not None
    layer = network.add_plugin_v3([plugin_input], [], plugin)
    for i in range(layer.num_outputs):
        out = layer.get_output(i)
        out.name = ["image_out", "transform"][i]
        network.mark_output(out)
    profile = builder.create_optimization_profile()
    shapes = [(1, channels, 1, 1), (2, channels, 5, 7), (3, channels, 20, 20)]
    if layout:
        shapes = [(n, h, w, c) for n, c, h, w in shapes]
    profile.set_shape("image", *shapes)
    assert tuple(profile.get_shape("image")[1]) == shapes[1]
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26)
    plan = builder.build_serialized_network(network, config)
    if expect_failure:
        assert plan is None
        return
    assert plan is not None
    return bytes(plan), logger


def infer(engine, x):
    context = engine.create_execution_context()
    context.set_input_shape("image", x.shape)
    assert tuple(context.get_tensor_shape("image")) == x.shape
    memories, outputs = [], {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        arr = (
            x
            if name == "image"
            else np.empty(
                context.get_tensor_shape(name),
                dtype=trt.nptype(engine.get_tensor_dtype(name)),
            )
        )
        mem = DeviceMem(arr.nbytes)
        memories.append(mem)
        if name == "image":
            memcpy_host_to_device(mem.device_ptr, arr)
        else:
            outputs[name] = (arr, mem)
    assert context.execute_v2([mem.device_ptr for mem in memories])
    for arr, mem in outputs.values():
        memcpy_device_to_host(arr, mem.device_ptr)
    return {name: arr for name, (arr, _) in outputs.items()}


@pytest.mark.parametrize("dtype", [trt.float32, trt.float16, trt.uint8])
@pytest.mark.parametrize("layout", [0, 1])
@pytest.mark.parametrize(
    "interpolation,pad_mode,swap_rb,normalize",
    [(0, 0, 0, 0), (1, 0, 1, 1), (0, 1, 1, 0), (1, 1, 0, 1)],
)
def test_dynamic_letterbox(
    creator, dtype, layout, interpolation, pad_mode, swap_rb, normalize
):
    options = dict(
        interpolation=interpolation,
        pad_mode=pad_mode,
        swap_rb=swap_rb,
        normalize=normalize,
        pad_value=[12.0, 34.0, 56.0],
    )
    plan, logger = build(creator, dtype, layout, **options)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    rng = np.random.default_rng(4811)
    for shape in [(2, 3, 5, 7), (1, 3, 13, 4), (3, 3, 1, 1), (1, 3, 9, 11)]:
        x = rng.integers(0, 256, shape).astype(trt.nptype(dtype))
        if layout:
            x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        actual = infer(engine, x)
        expected, transform = reference(x, layout, **options)
        tol = (
            0.13
            if dtype == trt.float16 and not normalize
            else (0.001 if dtype == trt.float16 else 0.0001)
        )
        np.testing.assert_allclose(actual["image_out"], expected, atol=tol, rtol=1e-5)
        np.testing.assert_allclose(actual["transform"], transform, atol=1e-6, rtol=1e-6)


def test_without_transform(creator):
    plan, logger = build(creator, trt.float32, 0, return_transform=0)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    actual = infer(engine, np.ones((1, 3, 5, 7), np.float32))
    assert set(actual) == {"image_out"}
    expected, _ = reference(np.ones((1, 3, 5, 7)), 0, 1, 0, 0, 0)
    np.testing.assert_allclose(actual["image_out"], expected)


@pytest.mark.parametrize(
    "options",
    [
        {"output_h": 0},
        {"output_w": -1},
        {"interpolation": 2},
        {"layout": 2},
        {"pad_mode": 2},
        {"swap_rb": -1},
        {"normalize": 2},
        {"return_transform": 2},
        {"pad_value": []},
        {"pad_value": [np.nan]},
        {"unknown": 0},
    ],
)
def test_invalid_fields(creator, options):
    assert make_plugin(creator, **options) is None


def test_onnx_uint8_graph(creator):
    import onnx
    from onnx import TensorProto as T, helper as h

    graph = h.make_graph(
        [
            h.make_node("Cast", ["image"], ["floating"], to=T.FLOAT),
            h.make_node(
                "ResizeWithPadPlugin",
                ["floating"],
                ["image_out", "transform"],
                domain="trt.plugins",
                output_h=9,
                output_w=11,
                layout=1,
                normalize=1,
                swap_rb=1,
                pad_value=[12.0, 34.0, 56.0],
            ),
        ],
        "letterbox",
        [h.make_tensor_value_info("image", T.UINT8, [2, 5, 7, 3])],
        [
            h.make_tensor_value_info("image_out", T.FLOAT, [2, 3, 9, 11]),
            h.make_tensor_value_info("transform", T.FLOAT, [2, 4]),
        ],
    )
    model = h.make_model(
        graph, opset_imports=[h.make_opsetid("", 18), h.make_opsetid("trt.plugins", 1)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(model.SerializeToString()), [
        str(parser.get_error(i)) for i in range(parser.num_errors)
    ]
    config = builder.create_builder_config()
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    x = np.arange(2 * 5 * 7 * 3, dtype=np.uint8).reshape(2, 5, 7, 3)
    actual = infer(engine, x)
    expected, transform = reference(x, 1, 1, 0, 1, 1, [12.0, 34.0, 56.0])
    np.testing.assert_allclose(actual["image_out"], expected, atol=1e-6)
    np.testing.assert_allclose(actual["transform"], transform, atol=1e-6)


@pytest.mark.parametrize(
    "fields",
    [
        [],
        [("output_h", np.array([9], np.int32), trt.PluginFieldType.INT32)],
        [
            ("output_h", np.array([9.0], np.float32), trt.PluginFieldType.FLOAT32),
            ("output_w", np.array([11], np.int32), trt.PluginFieldType.INT32),
        ],
        [
            ("output_h", np.array([9, 10], np.int32), trt.PluginFieldType.INT32),
            ("output_w", np.array([11], np.int32), trt.PluginFieldType.INT32),
        ],
        [
            ("output_h", np.array([9], np.int32), trt.PluginFieldType.INT32),
            ("output_w", np.array([11], np.int32), trt.PluginFieldType.INT32),
            ("output_h", np.array([9], np.int32), trt.PluginFieldType.INT32),
        ],
    ],
)
def test_malformed_fields(creator, fields):
    fc = trt.PluginFieldCollection([trt.PluginField(*field) for field in fields])
    assert creator.create_plugin("bad", fc, trt.TensorRTPhase.BUILD) is None


@pytest.mark.parametrize(
    "channels,pad", [(1, [7.0]), (4, [1.0, 2.0, 3.0, 4.0]), (4, [17.0])]
)
@pytest.mark.parametrize("layout", [0, 1])
def test_non_rgb_channels(creator, channels, pad, layout):
    plan, logger = build(creator, trt.float32, layout, channels=channels, pad_value=pad)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    x = np.arange(2 * channels * 5 * 7, dtype=np.float32).reshape(2, channels, 5, 7)
    if layout:
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    actual = infer(engine, x)
    expected, transform = reference(x, layout, 1, 0, 0, 0, pad)
    np.testing.assert_allclose(actual["image_out"], expected, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(actual["transform"], transform, atol=1e-6)


@pytest.mark.parametrize(
    "channels,options",
    [(1, {"swap_rb": 1}), (4, {"swap_rb": 1}), (3, {"pad_value": [1.0, 2.0]})],
)
def test_invalid_channel_configuration(creator, channels, options):
    build(creator, trt.float32, 0, channels=channels, expect_failure=True, **options)


def test_cuda_graph_replay(creator):
    from common_runtime import cuda, cuda_call

    plan, logger = build(creator, trt.float32, 0, pad_value=[1.0, 2.0, 3.0])
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    x = np.arange(105, dtype=np.float32).reshape(1, 3, 5, 7)
    context.set_input_shape("image", x.shape)
    buffers, hosts = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        hosts[name] = (
            x
            if name == "image"
            else np.empty(context.get_tensor_shape(name), np.float32)
        )
        buffers[name] = DeviceMem(hosts[name].nbytes)
        context.set_tensor_address(name, buffers[name].device_ptr)
    memcpy_host_to_device(buffers["image"].device_ptr, x)
    stream = cuda_call(cuda.cuStreamCreate(0))
    graph = executable = None
    try:
        assert context.execute_async_v3(int(stream))
        cuda_call(cuda.cuStreamSynchronize(stream))
        cuda_call(
            cuda.cuStreamBeginCapture(
                stream, cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL
            )
        )
        assert context.execute_async_v3(int(stream))
        graph = cuda_call(cuda.cuStreamEndCapture(stream))
        executable = cuda_call(cuda.cuGraphInstantiate(graph, 0))
        for value in (x, x + 10):
            memcpy_host_to_device(buffers["image"].device_ptr, value)
            cuda_call(cuda.cuGraphLaunch(executable, stream))
            cuda_call(cuda.cuStreamSynchronize(stream))
            memcpy_device_to_host(hosts["image_out"], buffers["image_out"].device_ptr)
            expected, _ = reference(value, 0, 1, 0, 0, 0, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(hosts["image_out"], expected, atol=1e-4)
    finally:
        if executable is not None:
            cuda_call(cuda.cuGraphExecDestroy(executable))
        if graph is not None:
            cuda_call(cuda.cuGraphDestroy(graph))
        cuda_call(cuda.cuStreamDestroy(stream))
