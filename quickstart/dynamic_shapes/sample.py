# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export and verify a decoder frontend with ASR length T and prosody length 2T."""

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "samples/python"))
import common_runtime as common


class DecoderFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f0_conv = torch.nn.Conv1d(1, 1, 3, stride=2, padding=1, bias=False)
        self.n_conv = torch.nn.Conv1d(1, 1, 3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            self.f0_conv.weight.copy_(torch.tensor([[[0.25, 0.5, 0.25]]]))
            self.n_conv.weight.copy_(torch.tensor([[[0.5, -0.25, 0.125]]]))

    def forward(self, asr, f0, noise):
        return torch.cat(
            [asr, self.f0_conv(f0.unsqueeze(1)), self.n_conv(noise.unsqueeze(1))], dim=1
        )


def export_model(model, path, bad_dimension_names=False):
    prosody = "asr_len" if bad_dimension_names else "prosody_len"
    inputs = (torch.zeros(1, 512, 40), torch.zeros(1, 80), torch.zeros(1, 80))
    torch.onnx.export(
        model,
        inputs,
        str(path),
        dynamo=False,
        opset_version=17,
        input_names=["ASR", "F0_PRED", "N_PRED"],
        output_names=["FEATURES"],
        dynamic_axes={
            "ASR": {0: "batch", 2: "asr_len"},
            "F0_PRED": {0: "batch", 1: prosody},
            "N_PRED": {0: "batch", 1: prosody},
            "FEATURES": {0: "batch", 2: "asr_len"},
        },
    )


def build_engine(path, logger):
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError(
            "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("ASR", (1, 512, 28), (1, 512, 100), (1, 512, 1106))
    for name in ["F0_PRED", "N_PRED"]:
        profile.set_shape(name, (1, 56), (1, 200), (1, 2212))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(
            "Engine build failed; inspect the dimension constraints above"
        )
    return plan


def infer(engine, feeds):
    context = engine.create_execution_context()
    for name, array in feeds.items():
        if not context.set_input_shape(name, array.shape):
            raise RuntimeError(f"Could not set shape for {name}")
    if context.infer_shapes():
        raise RuntimeError("Input shapes are incomplete")
    inputs, outputs, bindings = common.allocate_buffers(engine, context=context)
    try:
        for index, pointer in enumerate(bindings):
            if not context.set_tensor_address(engine.get_tensor_name(index), pointer):
                raise RuntimeError("Could not bind tensor")
        input_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        for name, buffer in zip(input_names, inputs):
            common.memcpy_host_to_device(buffer.device_ptr, feeds[name])
        with common.CudaStreamContext() as stream:
            if not context.execute_async_v3(stream.stream):
                raise RuntimeError("Inference failed")
            stream.synchronize()
        common.memcpy_device_to_host(outputs[0].host, outputs[0].device_ptr)
        return (
            outputs[0].host.copy().reshape(tuple(context.get_tensor_shape("FEATURES")))
        )
    finally:
        common.free_buffers(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bad-dimension-names",
        action="store_true",
        help="Reproduce the erroneous equality between T and 2T",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = DecoderFrontend().eval()
    path = args.output_dir / "frontend.onnx"
    export_model(model, path, args.bad_dimension_names)
    logger = trt.Logger(trt.Logger.WARNING)
    plan = build_engine(path, logger)
    (args.output_dir / "frontend.engine").write_bytes(plan)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize engine")
    rng = np.random.default_rng(0)
    for length in [28, 29, 40, 100, 101, 1106]:
        feeds = {
            "ASR": rng.normal(size=(1, 512, length)).astype(np.float32),
            "F0_PRED": rng.normal(size=(1, 2 * length)).astype(np.float32),
            "N_PRED": rng.normal(size=(1, 2 * length)).astype(np.float32),
        }
        with torch.no_grad():
            expected = model(
                *(
                    torch.from_numpy(feeds[name])
                    for name in ["ASR", "F0_PRED", "N_PRED"]
                )
            ).numpy()
        actual = infer(engine, feeds)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        print(
            f"T={length}: shape={actual.shape}, max_abs_error={np.max(np.abs(actual - expected)):.8g} PASS"
        )


if __name__ == "__main__":
    main()
