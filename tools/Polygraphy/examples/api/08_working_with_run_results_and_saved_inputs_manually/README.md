# Working With Run Results And Saved Inputs Manually

## Introduction

Inference inputs and outputs from `Comparator.run` can be serialized and saved to JSON
files so they can be reused. Inputs are stored as `List[Dict[str, np.ndarray]]` while outputs
are stored in a `RunResults` object, which can keep track of the outputs of multiple runners
from multiple inference iterations.

Command-line tools providing `--save-inputs` and `--save-outputs` options generally use these formats.

Usually, you'll only use saved inputs or `RunResults` with other Polygraphy APIs or
tools (as in [this example](../../cli//run/06_comparing_with_custom_output_data/)
or [this one](../../cli/inspect/05_inspecting_inference_outputs/)), but sometimes,
you may want to work with the underlying NumPy arrays manually.

Polygraphy includes convenience APIs that make it easy to load and manipulate these objects.

This example illustrates how you can load saved inputs and/or `RunResults` from a file
using the Python API and then access the NumPy arrays stored within.

## Reading NumPy Arrays From JSON

Use `RunResults.load("outputs.json")` for files produced by `--save-outputs`,
and `polygraphy.json.load_json("inputs.json")` for files produced by `--save-inputs`.
These APIs decode the arrays and restore their shape and dtype. For example,
to inspect every saved output without printing all its values:

```python
from polygraphy.comparator import RunResults

results = RunResults.load("outputs.json")
for runner_name, iterations in results.items():
    for iteration_index, outputs in enumerate(iterations):
        for name, array in outputs.items():
            print(runner_name, iteration_index, name, array.shape, array.dtype)
            if array.size:
                print("Range:", array.min(), array.max())
```

For a command-line summary, use `polygraphy inspect data outputs.json`.
Add `--show-values` only when you also want to print the tensor values.

The NumPy array's base64 payload contains a complete `.npy` file written by
`numpy.save`, including its header. It is not a buffer of raw tensor values.
Decoding the base64 string and passing all its bytes to `numpy.frombuffer`
interprets the header as data and loses the array's shape. For example, a
`float32` tensor of shape `(1, 2, 640, 640)` contains 819200 elements; a
128-byte header would add 32 spurious elements to that incorrect decoding,
potentially including very large numbers.

Do not remove a fixed number of header bytes: the
[NumPy file format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
has a variable-length header. Use the Polygraphy loading APIs above instead
of depending on the JSON encoding's internal fields. Loading restores the
stored values; it does not normalize or constrain the model's output range.

## Running The Example

1. Generate some inference inputs and outputs:

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --save-inputs inputs.json --save-outputs outputs.json
    ```

2. **[Optional]** Use `inspect data` to view the inputs on the command-line:

    ```bash
    polygraphy inspect data inputs.json --show-values
    ```

3. **[Optional]** Use `inspect data` to view the outputs on the command-line:

    ```bash
    polygraphy inspect data outputs.json --show-values
    ```

4. Run the example:

    ```bash
    python3 example.py
    ```
