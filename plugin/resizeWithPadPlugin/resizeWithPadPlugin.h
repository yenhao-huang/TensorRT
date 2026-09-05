// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRT_RESIZE_WITH_PAD_PLUGIN_H
#define TRT_RESIZE_WITH_PAD_PLUGIN_H

#include "NvInfer.h"
#include <array>
#include <string>
#include <vector>

namespace nvinfer1::plugin
{
struct ResizeWithPadConfig
{
    // output_h, output_w, interpolation, layout, pad_mode, swap_rb, normalize, return_transform
    std::array<int32_t, 8> values{0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<float> pad{114.0F};
};

class ResizeWithPadPlugin : public IPluginV3,
                            public IPluginV3OneCore,
                            public IPluginV3OneBuild,
                            public IPluginV3OneRuntime
{
public:
    explicit ResizeWithPadPlugin(ResizeWithPadConfig config, std::string pluginNamespace = "");
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    bool validShape(Dims const& dims, bool dynamic) const noexcept;
    ResizeWithPadConfig mConfig;
    std::string mNamespace;
    std::array<PluginField, 9> mFields;
    PluginFieldCollection mFC{};
};

class ResizeWithPadPluginCreator : public IPluginCreatorV3One
{
public:
    ResizeWithPadPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mNamespace;
    std::array<PluginField, 9> mFields;
    PluginFieldCollection mFC{};
};

int32_t launchResizeWithPad(void const* input, void* output, float* transform, float const* pad, int32_t padCount,
    DataType type, int32_t n, int32_t c, int32_t h, int32_t w, std::array<int32_t, 8> const& config,
    cudaStream_t stream) noexcept;
} // namespace nvinfer1::plugin
#endif
