# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from polygraphy.mod.trt_importer import lazy_import_trt

trt = lazy_import_trt()

from polygraphy.backend.trt import TrtRunner
from polygraphy.common import FormattedArray
from polygraphy.comparator import Comparator, RunResults
from polygraphy.datatype import DataType


@pytest.fixture(params=["LINEAR", "HWC8", "CHW2", "HWC16", "DHWC8"])
def vector_engine(request):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    rank = 5 if request.param == "DHWC8" else 4
    x = network.add_input("x", trt.float16, (-1,) * rank)
    y = network.add_identity(x).get_output(0)
    y.name = "y"
    network.mark_output(y)
    y.allowed_formats = 1 << int(getattr(trt.TensorFormat, request.param))
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "x",
        (1, 3) + (2,) * (rank - 2),
        (2, 10) + (3,) * (rank - 2),
        (3, 17) + (5,) * (rank - 2),
    )
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine.get_tensor_format("y") == getattr(trt.TensorFormat, request.param)
    yield engine, rank, request.param


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_logical_output(vector_engine, backend):
    engine, rank, _ = vector_engine
    with TrtRunner(engine) as runner:
        for n, c, spatial in [(3, 17, 5), (1, 3, 2), (2, 10, 3)]:
            shape = (n, c) + (spatial,) * (rank - 2)
            inputs = (
                (np.arange(np.prod(shape)) % 127 - 63).astype(np.float16).reshape(shape)
            )
            feed = (
                inputs
                if backend == "numpy"
                else pytest.importorskip("torch").from_numpy(inputs)
            )
            outputs = runner.infer({"x": feed})
            actual = outputs["y"] if backend == "numpy" else outputs["y"].numpy()
            assert actual.dtype == inputs.dtype
            np.testing.assert_array_equal(actual, inputs)
            results = RunResults()
            results.add([outputs], runner_name="vectorized")
            assert Comparator.validate(results, check_inf=True, check_nan=True)
            results.add([{"y": inputs}], runner_name="reference")
            assert Comparator.compare_accuracy(results)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_output(vector_engine, value):
    engine, rank, _ = vector_engine
    inputs = np.ones((1, 3) + (2,) * (rank - 2), dtype=np.float16)
    inputs.flat[-1] = value
    with TrtRunner(engine) as runner:
        results = RunResults()
        results.add([runner.infer({"x": inputs})], runner_name="nonfinite")
        assert not Comparator.validate(results, check_inf=True, check_nan=True)


@pytest.mark.parametrize("host", [True, False])
def test_raw_output(vector_engine, host):
    engine, rank, _ = vector_engine
    inputs = np.ones((1, 3) + (2,) * (rank - 2), dtype=np.float16)
    with TrtRunner(engine) as runner:
        outputs = runner.infer(
            {"x": inputs}, return_raw_buffers=True, copy_outputs_to_host=host
        )
        assert isinstance(outputs["y"], FormattedArray)
        assert tuple(outputs["y"].shape) == inputs.shape


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_padding_is_excluded(backend):
    from polygraphy.backend.trt.runner import _unpack_vectorized_output

    # NHWC with C=3 padded to 8, plus unused line padding.
    physical = np.full((1, 2, 4, 8), np.nan, dtype=np.float16)
    expected = np.arange(18, dtype=np.float16).reshape(1, 3, 2, 3)
    physical[:, :, :3, :3] = expected.transpose(0, 2, 3, 1)
    raw = physical.view(np.uint8).reshape(-1)
    if backend == "torch":
        raw = pytest.importorskip("torch").from_numpy(raw)
    actual = _unpack_vectorized_output(
        raw, DataType.FLOAT16, expected.shape, (8, 1, 4, 1), 1, 8
    )
    if backend == "torch":
        actual = actual.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_device_output_stays_formatted(vector_engine):
    engine, rank, format_name = vector_engine
    inputs = np.ones((1, 3) + (2,) * (rank - 2), dtype=np.float16)
    with TrtRunner(engine) as runner:
        output = runner.infer({"x": inputs}, copy_outputs_to_host=False)["y"]
        assert isinstance(output, FormattedArray) == (format_name != "LINEAR")


def test_unpack_buffer_bounds():
    from polygraphy.backend.trt.runner import _unpack_vectorized_output
    from polygraphy.exception import PolygraphyException

    with pytest.raises(PolygraphyException, match="smaller than"):
        _unpack_vectorized_output(
            np.empty(1, dtype=np.uint8),
            DataType.FLOAT16,
            (1, 3, 2, 3),
            (6, 1, 3, 1),
            1,
            8,
        )


def test_unpack_empty():
    from polygraphy.backend.trt.runner import _unpack_vectorized_output

    output = _unpack_vectorized_output(
        np.empty(0, dtype=np.uint8), DataType.FLOAT16, (0, 3, 2, 3), (6, 1, 3, 1), 1, 8
    )
    assert output.shape == (0, 3, 2, 3)
    assert output.dtype == np.float16
