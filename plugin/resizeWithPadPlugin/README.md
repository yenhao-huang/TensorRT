# ResizeWithPadPlugin

`ResizeWithPadPlugin` version `1` is an IPluginV3 implementation of aspect-ratio-preserving image resize with constant padding. It performs resize, channel swap, normalization, and NCHW output packing in one CUDA kernel. Output height and width are fixed plugin fields; input batch, height, width, and channel count may be dynamic.

TensorRT 11.2 does **not** allow UINT8 plugin tensor formats. For a UINT8 network input, insert an explicit GPU Cast to FLOAT before this plugin (examples below). The plugin itself supports FLOAT and HALF. This adds an intermediate floating-point tensor and does not provide the direct UINT8 plugin interface proposed in NVIDIA/TensorRT#4811.

## Interface

Input: one rank-4 NCHW or NHWC tensor. Output 0: NCHW with the same floating-point datatype as the plugin input. If enabled, output 1 is FLOAT `[N,4]` containing `(scale, pad_top, pad_left, 0)` for each image. All tensors use LINEAR storage. Nonpositive runtime dimensions are rejected. `swap_rb` requires three channels.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| output_h | INT32 scalar | required | Positive canvas height |
| output_w | INT32 scalar | required | Positive canvas width |
| interpolation | INT32 scalar | 1 | 0 nearest, 1 linear |
| layout | INT32 scalar | 0 | 0 NCHW input, 1 NHWC input |
| pad_mode | INT32 scalar | 0 | 0 centered, 1 top-left image placement |
| pad_value | FLOAT32 array | [114] | Scalar broadcast or one value per output channel |
| swap_rb | INT32 scalar | 0 | Reverse the three source channels |
| normalize | INT32 scalar | 0 | Divide the entire output, including padding, by 255 |
| return_transform | INT32 scalar | 1 | Enable the second output |

Unknown, duplicate, missing required, invalid enum, wrongly typed or nonscalar integer fields are rejected. Pad values must be nonempty and finite. Field values are serialized with the engine.

`scale = min(output_h / input_h, output_w / input_w)`. Resized dimensions use round-half-up, bounded to `[1, output_dimension]`. Center placement puts `floor((output_dimension - resized_dimension)/2)` pixels before the image; an odd extra padding pixel goes after it. Nearest uses asymmetric floor coordinates. Linear uses half-pixel coordinates and clamps samples to image edges, without antialiasing. Pad colors are specified in output channel order and are not swapped.

The transform reports the theoretical common scale, as in letterbox bounding-box remapping:

```python
original_x = (padded_x - pad_left) / scale
original_y = (padded_y - pad_top) / scale
```

Integer rounding of resized dimensions can make their realized ratios differ slightly from this common scale. Use the documented integer dimensions if exact pixel-center sampling coordinates are needed. Inputs with a different source size for each batch member must be grouped or processed separately, since one dense tensor has one shared H/W.

## TensorRT Python construction

After building the OSS plugin library and calling `trt.init_libnvinfer_plugins(logger, "")`:

```python
import numpy as np
import tensorrt as trt

image = network.add_input("image", trt.uint8, (-1, -1, -1, 3))
floating = network.add_cast(image, trt.float32).get_output(0)
creator = trt.get_plugin_registry().get_creator("ResizeWithPadPlugin", "1", "")
values = {"output_h": 640, "output_w": 640, "layout": 1, "normalize": 1}
arrays = {name: np.array([value], np.int32) for name, value in values.items()}
fields = trt.PluginFieldCollection([
    trt.PluginField(name, array, trt.PluginFieldType.INT32) for name, array in arrays.items()
])
plugin = creator.create_plugin("letterbox", fields, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3([floating], [], plugin)
preprocessed = layer.get_output(0)  # connect to the vision model
transform = layer.get_output(1)
```

Set an optimization profile covering the intended input batch/height/width before building a dynamic engine. For floating-point network input, connect it directly to the plugin. Keep the plugin library available when deserializing an engine.

## ONNX construction

The ONNX parser can import a custom `ResizeWithPadPlugin` node after registering the creator. For a UINT8 graph input, include the preceding Cast:

```python
from onnx import helper, TensorProto
nodes = [
    helper.make_node("Cast", ["image"], ["floating"], to=TensorProto.FLOAT),
    helper.make_node("ResizeWithPadPlugin", ["floating"], ["preprocessed", "transform"],
                     domain="trt.plugins", output_h=640, output_w=640,
                     layout=1, normalize=1, pad_value=[114.0]),
]
# Import the default ONNX domain and trt.plugins version 1 in the model.
```

Use one node output and `return_transform=0` when metadata is not needed. This custom operator requires this TensorRT plugin; ONNX Runtime cannot execute it without a separate implementation.

## Focused GPU tests

The normal OSS build includes this plugin via `plugin/CMakeLists.txt` and registers its creator in `plugin/api/inferPlugin.cpp`. A smaller test library compiles the same implementation and uses a test registration translation unit:

```bash
cmake -S plugin/resizeWithPadPlugin/tests -B /data/tenserrt/builds/4811/test \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNVINFER_LIBRARY=/path/to/libnvinfer.so.11
cmake --build /data/tenserrt/builds/4811/test -j2
RESIZE_WITH_PAD_LIBRARY=/data/tenserrt/builds/4811/test/libresize_with_pad_test.so \
  python -m pytest plugin/resizeWithPadPlugin/tests/test_resize_with_pad.py -q
```

Use matching TensorRT headers and library (this tree is 11.2.1), CUDA development files, Python bindings, NumPy, pytest, ONNX, and CUDA Python. Tests require a CUDA GPU; they do not silently skip GPU validation. The test library is for isolated validation and should not be loaded alongside another library registering the same creator.
