// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "resizeWithPadPlugin.h"
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nvinfer1::plugin
{
namespace
{
template <typename T>
__device__ float readPixel(
    T const* input, int64_t batch, int32_t channel, int32_t y, int32_t x, int32_t c, int32_t h, int32_t w, bool nhwc)
{
    int64_t const offset = nhwc ? ((batch * h + y) * w + x) * c + channel : ((batch * c + channel) * h + y) * w + x;
    return static_cast<float>(input[offset]);
}

template <typename T, typename U>
__global__ void resizeKernel(T const* input, U* output, float* transform, float const* pad, int32_t padCount,
    int64_t count, int32_t c, int32_t h, int32_t w, int32_t oh, int32_t ow, bool linear, bool nhwc, bool topLeft,
    bool swapRB, bool normalize)
{
    double const scale = fmin(static_cast<double>(oh) / h, static_cast<double>(ow) / w);
    int32_t const rh = max(1, min(oh, static_cast<int32_t>(fmin(static_cast<double>(oh), floor(h * scale + 0.5)))));
    int32_t const rw = max(1, min(ow, static_cast<int32_t>(fmin(static_cast<double>(ow), floor(w * scale + 0.5)))));
    int32_t const top = topLeft ? 0 : (oh - rh) / 2;
    int32_t const left = topLeft ? 0 : (ow - rw) / 2;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
        i += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        int32_t const x = i % ow;
        int32_t const y = (i / ow) % oh;
        int32_t const channel = (i / (static_cast<int64_t>(ow) * oh)) % c;
        int64_t const batch = i / (static_cast<int64_t>(ow) * oh * c);
        int32_t const sourceChannel = swapRB ? 2 - channel : channel;
        // pad_value is specified in output channel order.
        float value = pad[padCount == 1 ? 0 : channel];
        if (y >= top && y < top + rh && x >= left && x < left + rw)
        {
            if (linear)
            {
                float const sy = fmaxf(0.F, (y - top + 0.5F) * h / rh - 0.5F);
                float const sx = fmaxf(0.F, (x - left + 0.5F) * w / rw - 0.5F);
                int32_t const y0 = min(h - 1, static_cast<int32_t>(floorf(sy)));
                int32_t const x0 = min(w - 1, static_cast<int32_t>(floorf(sx)));
                int32_t const y1 = min(h - 1, y0 + 1);
                int32_t const x1 = min(w - 1, x0 + 1);
                float const dy = sy - y0, dx = sx - x0;
                float const a = readPixel(input, batch, sourceChannel, y0, x0, c, h, w, nhwc);
                float const b = readPixel(input, batch, sourceChannel, y0, x1, c, h, w, nhwc);
                float const d = readPixel(input, batch, sourceChannel, y1, x0, c, h, w, nhwc);
                float const e = readPixel(input, batch, sourceChannel, y1, x1, c, h, w, nhwc);
                value = (a + (b - a) * dx) * (1.F - dy) + (d + (e - d) * dx) * dy;
            }
            else
            {
                int32_t const sy = min(h - 1, static_cast<int32_t>(static_cast<int64_t>(y - top) * h / rh));
                int32_t const sx = min(w - 1, static_cast<int32_t>(static_cast<int64_t>(x - left) * w / rw));
                value = readPixel(input, batch, sourceChannel, sy, sx, c, h, w, nhwc);
            }
        }
        output[i] = static_cast<U>(normalize ? value / 255.F : value);
        if (transform && channel == 0 && y == 0 && x == 0)
        {
            transform[batch * 4] = scale;
            transform[batch * 4 + 1] = top;
            transform[batch * 4 + 2] = left;
            transform[batch * 4 + 3] = 0.F;
        }
    }
}
} // namespace
int32_t launchResizeWithPad(void const* input, void* output, float* transform, float const* pad, int32_t padCount,
    DataType type, int32_t n, int32_t c, int32_t h, int32_t w, std::array<int32_t, 8> const& p,
    cudaStream_t stream) noexcept
{
    int64_t const count = static_cast<int64_t>(n) * c * p[0] * p[1];
    int32_t const blocks = static_cast<int32_t>(std::min<int64_t>((count + 255) / 256, 65535));
#define LAUNCH(T, U)                                                                                                   \
    resizeKernel<<<blocks, 256, 0, stream>>>(static_cast<T const*>(input), static_cast<U*>(output), transform, pad,    \
        padCount, count, c, h, w, p[0], p[1], p[2] == 1, p[3] == 1, p[4] == 1, p[5] != 0, p[6] != 0)
    switch (type)
    {
    case DataType::kHALF: LAUNCH(__half, __half); break;
    case DataType::kFLOAT: LAUNCH(float, float); break;
    default: return -1;
    }
#undef LAUNCH
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
} // namespace nvinfer1::plugin
