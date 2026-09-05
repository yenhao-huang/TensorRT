# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs
import pytest
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import build_anchor_generator

from create_onnx import DET2GraphSurgeon
from image_batcher import ImageBatcher


@pytest.mark.parametrize("height", [0, 769, "height"])
def test_exported_resolution_is_fixed_and_divisible(height):
    surgeon = DET2GraphSurgeon.__new__(DET2GraphSurgeon)
    surgeon.graph = gs.Graph(inputs=[gs.Variable("image", shape=[3, height, 768])])
    with pytest.raises(ValueError, match="positive fixed multiples of 32"):
        surgeon.update_preprocessor(1)


@pytest.fixture
def model_config():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 768
    return cfg


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (640, 320), (200, 120, 40)).save(path)
    return str(path)


@pytest.mark.parametrize("height,width", [(768, 768), (1344, 1344), (768, 1024)])
def test_anchors_match_exported_resolution(model_config, sample_image, height, width):
    # Run the real Detectron2 backbone and anchor generator without downloading weights.
    # Anchor coordinates depend on feature-map geometry, not trained weight values.
    surgeon = DET2GraphSurgeon.__new__(DET2GraphSurgeon)
    surgeon.det2_cfg = model_config
    surgeon.height, surgeon.width = height, width
    with torch.no_grad():
        anchors = surgeon.get_anchors(sample_image)
    strides = [4, 8, 16, 32, 64]
    reference = build_anchor_generator(
        model_config, [ShapeSpec(stride=s) for s in strides]
    )
    feature_maps = [
        torch.zeros(1, 256, (height + s - 1) // s, (width + s - 1) // s)
        for s in strides
    ]
    expected = torch.cat([boxes.tensor for boxes in reference(feature_maps)]).numpy()
    np.testing.assert_array_equal(anchors, expected)


def test_anchor_resize_must_fit_exported_resolution(model_config, sample_image):
    model_config.INPUT.MIN_SIZE_TEST = 800
    model_config.INPUT.MAX_SIZE_TEST = 1333
    surgeon = DET2GraphSurgeon.__new__(DET2GraphSurgeon)
    surgeon.det2_cfg = model_config
    surgeon.height = surgeon.width = 768
    with pytest.raises(ValueError, match="exceeds.*768"):
        surgeon.get_anchors(sample_image)


def test_batcher_rejects_cropping(model_config, sample_image, tmp_path):
    model_config.INPUT.MIN_SIZE_TEST = 800
    model_config.INPUT.MAX_SIZE_TEST = 1333
    config_path = tmp_path / "model.yaml"
    config_path.write_text(model_config.dump())
    batcher = ImageBatcher(
        sample_image, [1, 3, 768, 768], np.float32, config_file=str(config_path)
    )
    with pytest.raises(ValueError, match="exceeds.*768"):
        batcher.preprocess_image(sample_image)


@pytest.mark.parametrize("side", [768, 1344])
def test_batcher_preserves_aspect_ratio(model_config, sample_image, tmp_path, side):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(model_config.dump())
    batcher = ImageBatcher(
        sample_image, [1, 3, side, side], np.float32, config_file=str(config_path)
    )
    image, scale = batcher.preprocess_image(sample_image)
    assert image.shape == (3, side, side)
    assert scale == 1.2
    np.testing.assert_array_equal(image[:, 0, 0], [40, 120, 200])
    np.testing.assert_array_equal(image[:, -1, -1], [104, 116, 124])
