# scatterElements

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Python API usage](#python-api-usage)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The scatterElements plugin implements the scatter operation described in (https://github.com/rusty1s/pytorch_scatter), in compliance with the [ONNX specification for ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements)

Note: ScatterElements with reduce="none" is implemented in TRT core, not this plugin.

### Structure

This plugin has the 2 versions. The latest is plugin creator class `ScatterElementsPluginV3Creator` and the plugin class `ScatterElementsPluginV3` which extends `IPluginV3`. (name: `ScatterElements`, version: 2)
The legacy plugin that will be deprecated, is plugin creator class `ScatterElementsPluginV2Creator` and the plugin class `ScatterElementsPluginV2`, which extends `IPluginV2DynamicExt` (name: `ScatterElements`, version: 1).

The `ScatterElements` plugin consumes the following inputs:

1. `data` - T: Tensor of rank r >= 1.
2. `indices` - Tind: Tensor of int64 indices, of r >= 1 (same rank as input). All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.
3. `updates` - T: Tensor of rank r >=1 (same rank and shape as indices)

The `ScatterElements` plugin produces the following output:

1. `output` - T: Tensor, same shape as `data`.

## Python API usage

In TensorRT 10.6 and later, select plugin version `"2"` to create the
`IPluginV3` implementation. Initialize the standard plugin library before
looking up the creator. The example below builds a scatter-add network directly
with the Python API, without exporting an ONNX model:

```python
import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
creator = trt.get_plugin_registry().get_creator("ScatterElements", "2", "")
if creator is None:
    raise RuntimeError("ScatterElements version 2 is not registered")

# Keep the field values alive until create_plugin returns.
axis = np.array([1], dtype=np.int32)
fields = trt.PluginFieldCollection([
    trt.PluginField("axis", axis, trt.PluginFieldType.INT32),
    trt.PluginField("reduction", b"add", trt.PluginFieldType.CHAR),
])
plugin = creator.create_plugin("scatter_add", fields, trt.TensorRTPhase.BUILD)
if plugin is None:
    raise RuntimeError("Could not create ScatterElements plugin")

builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
)
data = network.add_input("data", trt.float32, (3, 3))
indices = network.add_input("indices", trt.int64, (3, 2))
updates = network.add_input("updates", trt.float32, (3, 2))
layer = network.add_plugin_v3(
    inputs=[data, indices, updates], shape_inputs=[], plugin=plugin
)
output = layer.get_output(0)
output.name = "output"
network.mark_output(output)
config = builder.create_builder_config()
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise RuntimeError("Could not build scatter-add engine")
print(f"Built ScatterElements V3 engine: {plan.nbytes} bytes")
```

The input order is `data`, `indices`, `updates`; all three are execution inputs,
so `shape_inputs` is empty. This plugin requires `INT64` indices. The field
`axis` uses an `INT32` array, while `reduction` uses the bytes `b"add"` with
`PluginFieldType.CHAR`. Pass `TensorRTPhase.BUILD` to the V3 creator and use
`add_plugin_v3` to add the resulting plugin to the network.

The output starts with a copy of `data` and adds each update at its indexed
position. Repeated indices accumulate updates. For example, with `axis=1`,
`data=[[10, 20, 30], [40, 50, 60], [70, 80, 90]]`,
`indices=[[0, 0], [1, 2], [2, 2]]`, and
`updates=[[1, 2], [3, 4], [5, 6]]`, the output is
`[[13, 20, 30], [40, 53, 64], [70, 80, 101]]`.

Older released plugin libraries can mishandle negative indices or a negative
`axis`. Use nonnegative equivalents with those libraries, or rebuild the plugin
library from this source tree to use the corrected negative-index handling.

Register the same plugin library before deserializing the engine in another
process. If the creator lookup fails, verify that the installed plugin library
matches the TensorRT runtime and provides `ScatterElements` version `2`.

## Parameters

The `ScatterElements` plugin has the following parameters:

| Type             | Parameter                       | Description
|------------------|---------------------------------|--------------------------------------------------------
|`int`             |`axis`                           | Which axis to scatter on. Default is 0. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
|`char`            |`reduction`                      | Type of reduction to apply: add, mul, max, min. ‘add’: reduction using the addition operation. ‘mul’: reduction using the multiplication operation.‘max’: reduction using the maximum operation.‘min’: reduction using the minimum operation.


The following resources provide a deeper understanding of the `scatterElements` plugin:

- [pytorch_scatter: original implementation and docs](https://github.com/rusty1s/pytorch_scatter)
- [ONNX specification for ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog

- July 2024: Version 2 of the plugin migrated to `IPluginV3` interface design. The legacy plugin (version 1) using `IPluginV2DynamicExt` interface is deprecated.
- Oct 2023: This is the first release of this `README.md` file.

## Known issues

- Types T=BFLOAT16 and T=INT8 are currently not supported.
- ONNX spec allows Tind=int32 : only INT64 is supported by this plugin
