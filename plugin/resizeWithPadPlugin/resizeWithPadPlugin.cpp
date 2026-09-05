// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "resizeWithPadPlugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nvinfer1::plugin
{
namespace
{
constexpr char const* kName = "ResizeWithPadPlugin";
constexpr char const* kVersion = "1";
constexpr std::array<char const*, 8> kNames{
    "output_h", "output_w", "interpolation", "layout", "pad_mode", "swap_rb", "normalize", "return_transform"};
void require(bool condition, char const* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}
bool supportedType(DataType type)
{
    return type == DataType::kFLOAT || type == DataType::kHALF;
}
} // namespace
ResizeWithPadPlugin::ResizeWithPadPlugin(ResizeWithPadConfig config, std::string pluginNamespace)
    : mConfig(std::move(config))
    , mNamespace(std::move(pluginNamespace))
{
    require(mConfig.values[0] > 0 && mConfig.values[1] > 0, "output_h and output_w must be positive");
    for (size_t i = 2; i < mConfig.values.size(); ++i)
        require(mConfig.values[i] == 0 || mConfig.values[i] == 1, "enum and boolean fields must be 0 or 1");
    require(!mConfig.pad.empty(), "pad_value cannot be empty");
    for (float value : mConfig.pad)
        require(std::isfinite(value), "pad_value must be finite");
    for (size_t i = 0; i < kNames.size(); ++i)
        mFields[i] = PluginField{kNames[i], &mConfig.values[i], PluginFieldType::kINT32, 1};
    mFields[8] = PluginField{
        "pad_value", mConfig.pad.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mConfig.pad.size())};
    mFC = PluginFieldCollection{static_cast<int32_t>(mFields.size()), mFields.data()};
}
IPluginCapability* ResizeWithPadPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
    case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    default: return nullptr;
    }
}
IPluginV3* ResizeWithPadPlugin::clone() noexcept
{
    try
    {
        return new ResizeWithPadPlugin(mConfig, mNamespace);
    }
    catch (...)
    {
        return nullptr;
    }
}
char const* ResizeWithPadPlugin::getPluginName() const noexcept
{
    return kName;
}
char const* ResizeWithPadPlugin::getPluginVersion() const noexcept
{
    return kVersion;
}
char const* ResizeWithPadPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
int32_t ResizeWithPadPlugin::getNbOutputs() const noexcept
{
    return 1 + mConfig.values[7];
}

bool ResizeWithPadPlugin::validShape(Dims const& dims, bool dynamic) const noexcept
{
    if (dims.nbDims != 4)
        return false;
    int64_t volume = 1;
    for (int32_t i = 0; i < 4; ++i)
    {
        if (dynamic && dims.d[i] == -1)
            continue;
        if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max()
            || volume > std::numeric_limits<int64_t>::max() / dims.d[i])
            return false;
        volume *= dims.d[i];
    }
    if (volume > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float)))
        return false;
    int64_t const c = dims.d[mConfig.values[3] == 0 ? 1 : 3];
    if (c != -1)
    {
        if (mConfig.values[5] && c != 3)
            return false;
        if (mConfig.pad.size() != 1 && static_cast<int64_t>(mConfig.pad.size()) != c)
            return false;
    }
    int64_t outVolume = static_cast<int64_t>(mConfig.values[0]) * mConfig.values[1];
    for (int64_t dim : {dims.d[0], c})
    {
        if (dynamic && dim == -1)
            continue;
        if (dim <= 0 || outVolume > std::numeric_limits<int64_t>::max() / dim)
            return false;
        outVolume *= dim;
    }
    return outVolume <= std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float));
}
int32_t ResizeWithPadPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const*, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != getNbOutputs() || !supportedType(in[0].desc.type))
        return -1;
    return validShape(in[0].desc.dims, true) && validShape(in[0].min, false) && validShape(in[0].max, false) ? 0 : -1;
}
bool ResizeWithPadPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* io, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != getNbOutputs() || pos < 0 || pos >= nbInputs + nbOutputs)
        return false;
    if (io[pos].desc.format != TensorFormat::kLINEAR)
        return false;
    if (pos == 0)
        return supportedType(io[0].desc.type);
    return io[pos].desc.type == (pos == 1 ? io[0].desc.type : DataType::kFLOAT);
}
int32_t ResizeWithPadPlugin::getOutputDataTypes(
    DataType* types, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbInputs != 1 || nbOutputs != getNbOutputs() || !supportedType(inputTypes[0]))
        return -1;
    types[0] = inputTypes[0];
    if (nbOutputs == 2)
        types[1] = DataType::kFLOAT;
    return 0;
}
int32_t ResizeWithPadPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& builder) noexcept
{
    if (nbInputs != 1 || nbShapeInputs != 0 || nbOutputs != getNbOutputs() || inputs[0].nbDims != 4)
        return -1;
    outputs[0].nbDims = 4;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[mConfig.values[3] == 0 ? 1 : 3];
    outputs[0].d[2] = builder.constant(mConfig.values[0]);
    outputs[0].d[3] = builder.constant(mConfig.values[1]);
    if (nbOutputs == 2)
    {
        outputs[1].nbDims = 2;
        outputs[1].d[0] = inputs[0].d[0];
        outputs[1].d[1] = builder.constant(4);
    }
    return 0;
}
size_t ResizeWithPadPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    return mConfig.pad.size() * sizeof(float);
}
int32_t ResizeWithPadPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != getNbOutputs() || !validShape(in[0].dims, false) || !supportedType(in[0].type)
        || in[0].format != TensorFormat::kLINEAR)
        return -1;
    auto const& d = in[0].dims;
    auto const& o = out[0].dims;
    if (o.nbDims != 4 || o.d[0] != d.d[0] || o.d[1] != d.d[mConfig.values[3] == 0 ? 1 : 3]
        || o.d[2] != mConfig.values[0] || o.d[3] != mConfig.values[1] || out[0].type != in[0].type)
        return -1;
    if (nbOutputs == 2
        && (out[1].dims.nbDims != 2 || out[1].dims.d[0] != d.d[0] || out[1].dims.d[1] != 4
            || out[1].type != DataType::kFLOAT))
        return -1;
    return 0;
}
int32_t ResizeWithPadPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (!workspace || onShapeChange(inputDesc, 1, outputDesc, getNbOutputs()) != 0)
        return -1;
    if (cudaMemcpyAsync(
            workspace, mConfig.pad.data(), mConfig.pad.size() * sizeof(float), cudaMemcpyHostToDevice, stream)
        != cudaSuccess)
        return -1;
    auto const& d = inputDesc[0].dims;
    bool const nhwc = mConfig.values[3] == 1;
    return launchResizeWithPad(inputs[0], outputs[0], getNbOutputs() == 2 ? static_cast<float*>(outputs[1]) : nullptr,
        static_cast<float const*>(workspace), static_cast<int32_t>(mConfig.pad.size()), inputDesc[0].type, d.d[0],
        d.d[nhwc ? 3 : 1], d.d[nhwc ? 1 : 2], d.d[nhwc ? 2 : 3], mConfig.values, stream);
}
IPluginV3* ResizeWithPadPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}
PluginFieldCollection const* ResizeWithPadPlugin::getFieldsToSerialize() noexcept
{
    return &mFC;
}

ResizeWithPadPluginCreator::ResizeWithPadPluginCreator()
{
    for (size_t i = 0; i < kNames.size(); ++i)
        mFields[i] = PluginField{kNames[i], nullptr, PluginFieldType::kINT32, 1};
    mFields[8] = PluginField{"pad_value", nullptr, PluginFieldType::kFLOAT32, 1};
    mFC = PluginFieldCollection{static_cast<int32_t>(mFields.size()), mFields.data()};
}
char const* ResizeWithPadPluginCreator::getPluginName() const noexcept
{
    return kName;
}
char const* ResizeWithPadPluginCreator::getPluginVersion() const noexcept
{
    return kVersion;
}
char const* ResizeWithPadPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
void ResizeWithPadPluginCreator::setPluginNamespace(char const* ns) noexcept
{
    try
    {
        mNamespace = ns ? ns : "";
    }
    catch (...)
    {
    }
}
PluginFieldCollection const* ResizeWithPadPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
IPluginV3* ResizeWithPadPluginCreator::createPlugin(
    char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept
{
    try
    {
        require(fc && fc->nbFields >= 0 && (fc->nbFields == 0 || fc->fields), "invalid field collection");
        ResizeWithPadConfig config;
        std::array<bool, 9> seen{};
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            auto const& f = fc->fields[i];
            require(f.name && f.data, "field name/data cannot be null");
            auto const it = std::find_if(
                kNames.begin(), kNames.end(), [&](char const* n) { return std::strcmp(n, f.name) == 0; });
            size_t const index = it - kNames.begin();
            require(index != 8 || std::strcmp(f.name, "pad_value") == 0, "unknown field");
            require(!seen[index], "duplicate field");
            seen[index] = true;
            if (index == 8)
            {
                require(f.type == PluginFieldType::kFLOAT32 && f.length > 0, "pad_value must be nonempty float32");
                auto const* values = static_cast<float const*>(f.data);
                config.pad.assign(values, values + f.length);
            }
            else
            {
                require(f.type == PluginFieldType::kINT32 && f.length == 1, "integer field must be scalar int32");
                config.values[index] = *static_cast<int32_t const*>(f.data);
            }
        }
        require(seen[0] && seen[1], "output_h and output_w are required");
        return new ResizeWithPadPlugin(std::move(config), mNamespace);
    }
    catch (std::exception const& error)
    {
        std::fprintf(stderr, "ResizeWithPadPlugin: %s\n", error.what());
        return nullptr;
    }
    catch (...)
    {
        return nullptr;
    }
}
} // namespace nvinfer1::plugin
