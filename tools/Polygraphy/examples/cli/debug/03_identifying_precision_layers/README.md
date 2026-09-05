# Identifying Layers From A Precision Search

`polygraphy debug precision` searches for a range of layers to run in higher
precision, using your check command to decide whether each configuration is
acceptable. It requires TensorRT's weakly typed network support and precision
flags (for example, TensorRT 10.x); it is not applicable to strongly typed networks.

## Running The Search

With your ONNX model, saved inputs, and reference outputs:

```bash
polygraphy debug precision model.onnx --fp16 --mode bisect \
    --check polygraphy run polygraphy_debug.engine --trt \
    --load-inputs inputs.json --load-outputs reference.json \
    --atol 1e-3 --rtol 1e-3
```

The final summary reports a successful range such as "first 1301 layers".
This refers to indices 0 through 1300 in the **parsed TensorRT network**.
With `--direction reverse`, it refers to the last N layers instead.
The count includes layers that cannot have their precision changed, such as
constants and shape operations.

The summary then lists the layers actually marked in that successful range,
including their indices, names, input tensors, and output tensors. For example:

```text
Higher-precision layers in the successful configuration (1 marked, DataType.FLOAT):
Indices refer to the parsed TensorRT network, not ONNX node indices.
Layer 0    | square_sensitive [Op: LayerType.ELEMENTWISE]
    {X [dtype=float32, shape=(1, 1, 16, 16)],
     X [dtype=float32, shape=(1, 1, 16, 16)]}
     -> {square [dtype=float32, shape=(1, 1, 16, 16)]}
```

This list comes from the successful search boundary, even if the last tested
configuration failed. The tensor dtypes shown describe the parsed network's
inputs and outputs; they are not a dump of the engine's internal compute types.

## Relating The Result To ONNX And Reusing It

Inspect both representations to follow the named tensors:

```bash
polygraphy inspect model model.onnx --show layers
polygraphy inspect model model.onnx --display-as trt --show layers
```

Use the reported layer names and tensor connections to locate the corresponding
ONNX operations. Parsing can introduce or transform layers, so TensorRT layer
indices do not identify ONNX nodes one-to-one. Engine optimization can further
fuse layers; engine layer numbers are a different numbering scheme as well.

For example, if the report identifies `square_sensitive`, rebuild and check with:

```bash
polygraphy run model.onnx --trt --fp16 \
    --layer-precisions square_sensitive:float32 --precision-constraints obey \
    --load-inputs inputs.json --load-outputs reference.json \
    --atol 1e-3 --rtol 1e-3
```

Replace the example name with all the marked layer names from your report.
Quote each `name:float32` argument if its name contains spaces or shell characters.
Use the same model, parser options, and any preexisting precision constraints as
the search. For networks with duplicate layer names, use a network configuration
script and the reported indices to disambiguate them.

The result passed the supplied check on the supplied data; it does not prove
accuracy on other inputs or that every listed layer individually needs higher
precision. Validate the rebuilt configuration on representative data. The last
intermediate engine may be from a failed iteration, so rebuild from the reported
configuration instead of assuming that engine is the successful one.
