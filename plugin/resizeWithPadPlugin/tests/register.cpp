// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "resizeWithPadPlugin.h"
using nvinfer1::plugin::ResizeWithPadPluginCreator;
REGISTER_TENSORRT_PLUGIN(ResizeWithPadPluginCreator);
