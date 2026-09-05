# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
from infer import TensorRTInfer


def test_infer_uses_owned_device_memory(tmp_path):
    """Exercise allocation, execution, copies, and result parsing on a real GPU."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    image = network.add_input("image", trt.float32, [1, 3, 32, 32])
    average = network.add_reduce(image, trt.ReduceOperation.AVG, 15, False)
    score = network.add_shuffle(average.get_output(0))
    score.reshape_dims = (1, 1)
    arrays = [
        ("num_detections", np.array([[1]], dtype=np.int32)),
        ("boxes", np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)),
        ("scores", None),
        ("classes", np.array([[16]], dtype=np.int32)),
        ("masks", np.ones((1, 1, 28, 28), dtype=np.float32)),
    ]
    for name, array in arrays:
        output = (
            score.get_output(0)
            if array is None
            else network.add_constant(array.shape, array).get_output(0)
        )
        output.name = name
        network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine_path = tmp_path / "smoke.trt"
    engine_path.write_bytes(bytes(plan))
    runner = TensorRTInfer(engine_path)
    for value in (1.0, 0.75):
        detections = runner.infer(
            np.full((1, 3, 32, 32), value, dtype=np.float32), [1.0]
        )
        assert len(detections) == 1 and len(detections[0]) == 1
        detection = detections[0][0]
        assert detection["class"] == 16
        assert detection["score"] == pytest.approx(value)
        assert detection["xmin"] == pytest.approx(6.4)
        assert detection["ymax"] == pytest.approx(9.6)
        np.testing.assert_array_equal(detection["mask"], np.ones((28, 28)))
