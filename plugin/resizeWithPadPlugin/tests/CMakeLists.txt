# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(resize_with_pad_test_plugin LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIBRARY nvinfer HINTS "${TRT_LIB_DIR}" REQUIRED)
add_library(resize_with_pad_test SHARED ../resizeWithPadPlugin.cpp ../resizeWithPadKernel.cu register.cpp)
target_compile_features(resize_with_pad_test PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(resize_with_pad_test PRIVATE .. "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")
target_link_libraries(resize_with_pad_test PRIVATE "${NVINFER_LIBRARY}" CUDA::cudart)
