# Exporting inputs with related dynamic lengths

This example explains a decoder export failure where ASR features have length
`T`, but pitch and noise inputs have length `2*T`. Giving all three dimensions
the same ONNX name incorrectly declares them equal. TensorRT uses dimension
names as equality constraints, including names imported from ONNX.
See [Named Dimensions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html).

## The failing pattern

The frontend takes `ASR[B,512,T]`, `F0_PRED[B,P]`, and `N_PRED[B,P]`. Each
prosody input passes through a 1D convolution with kernel 3, stride 2, and
padding 1. Its output length is `ceil(P/2)`. Concatenating those outputs with
ASR along the channel dimension requires `T == ceil(P/2)`.

For the inputs used here, `P=2*T`, so the convolution output has length `T`.
If ASR and prosody both use the dimension name `seq_len`, TensorRT may substitute
`T` for `P`. At the optimization point `T=100`, it then sees a Concat of lengths
100 and 50 and rejects the engine. Increasing workspace or changing precision
does not correct this contradictory shape annotation.

## Correct the export

Use a shared name only for dimensions that must actually be equal. For the
full decoder pattern in [issue #4682](https://github.com/NVIDIA/TensorRT/issues/4682),
replace its `dynamic_axes` dictionary with:

```python
dynamic_axes = {
    "ASR": {0: "batch", 2: "asr_len"},
    "F0_PRED": {0: "batch", 1: "prosody_len"},
    "N_PRED": {0: "batch", 1: "prosody_len"},
    "REF": {0: "batch"},
    "AUDIO_OUT": {0: "batch", 2: "audio_len"},
}
```

Pass this dictionary to the original `torch.onnx.export` call and re-export
the model. The full decoder's audio length also needs its own name: the
upsampling generator does not generally return a sequence of length `T`.
Batch dimensions remain equal, and F0/noise lengths remain equal.

The executable sample uses `dynamo=False` explicitly to reproduce the
`dynamic_axes` exporter path in the reported code. With the newer
`dynamo=True` exporter, use `dynamic_shapes` instead; do not assume
`dynamic_axes` provides the same shape contract.
See [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx_export.html).

The sample exports only the frontend, whose `FEATURES` output really does have
length `T`, so that output correctly shares the `asr_len` name. It is not a
complete vocoder and does not produce audio.

## Keep the profile and runtime shapes consistent

The original profile bounds can remain:

| Input | Minimum | Optimum | Maximum |
| --- | --- | --- | --- |
| ASR | (1,512,28) | (1,512,100) | (1,512,1106) |
| F0_PRED | (1,56) | (1,200) | (1,2212) |
| N_PRED | (1,56) | (1,200) | (1,2212) |
| REF, in the full decoder | (1,128) | (1,128) | (1,128) |

Different names remove a false equality; they do not encode the relationship
`P=2*T`. The application must continue supplying compatible lengths at runtime.
A profile's min/max bounds do not make every combination inside those bounds
valid. This example always supplies `P=2*T`. The frontend convolution also
admits `P=2*T-1`, but the complete decoder may impose additional constraints.

## Run the example

Install a TensorRT runtime appropriate for your GPU and the Python dependencies:

```bash
python -m pip install -r quickstart/dynamic_shapes/requirements.txt
python quickstart/dynamic_shapes/sample.py --output-dir /tmp/trt-dynamic-shapes
```

The script exports a deterministic, untrained PyTorch frontend, builds and
serializes an FP32 strongly typed engine, deserializes it, and compares GPU
outputs with PyTorch CPU outputs for `T=28,29,40,100,101,1106`. It writes
`frontend.onnx` and `frontend.engine` to the output directory and prints `PASS`
for each length. It checks the min/opt/max profile points and additional lengths;
it does not measure vocoder quality or benchmark performance.

To reproduce the incorrect annotation, use a separate output directory:

```bash
python quickstart/dynamic_shapes/sample.py --bad-dimension-names \
  --output-dir /tmp/trt-dynamic-shapes-bad
```

This command is expected to fail during engine building with a shape constraint
error containing `100 != 50`. The network operations and weights are identical;
only the names assigned to dynamic input dimensions differ.
