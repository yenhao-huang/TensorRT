# Introductory Notebooks

Notebook 2 chooses FP32 or FP16 in PyTorch **before ONNX export** and builds a
strongly typed TensorRT network. Runtime buffers must match the engine's I/O
types; changing a NumPy dtype does not change an engine's compute precision.
Rerun export and build after changing `USE_FP16` or `BATCH_SIZE`.

The `convert_onnx_to_engine` helper preserves its `fp16_mode` argument:

- `False` builds the ONNX graph with its existing types. Use this for the
  notebook's exported FP32/FP16 model or an explicitly mixed-precision model.
- `True` (the default) first converts an FP32 ONNX graph with
  `onnxconverter-common`, preserving its I/O types. Install the optional
  conversion dependencies with `python -m pip install -r requirements.txt`.
  This uses explicit FP16 operations and casts, rather than a builder flag.
  Already converted models should use `False` to avoid converting them twice.

TensorRT and CUDA Python must also be installed for your CUDA environment.
The notebooks additionally use PyTorch, torchvision, NumPy, scikit-image,
matplotlib, and Jupyter. The Python builder uses strong typing on TensorRT 10.x
and 11.x; these notebooks do not depend on the removed FP16 builder flag.
The helper disables TF32 to make the FP32 reference comparison reproducible
without TF32 rounding; this can affect performance.

Explicit conversion can change numerical results and is not equivalent to the
old builder's choice of lower-precision tactics. The helper uses the FP16
representable constant bounds (smallest subnormal through largest finite value);
values outside those bounds can change, and some operations remain FP32 under
the converter's default blocklist. See the
[ONNX Runtime conversion documentation](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).
For selective FP32 operations, see the
[mixed-precision autocast sample](../../samples/python/strongly_type_autocast/).

Compare outputs on the same normalized inputs, then evaluate representative
data before deploying. Notebook 2 reports logits and checks numerical agreement;
its repeated single-image batch is not a dataset accuracy benchmark.
