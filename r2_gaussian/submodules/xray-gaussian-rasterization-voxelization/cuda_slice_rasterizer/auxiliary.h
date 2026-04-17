/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_SLICE_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_SLICE_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <stdexcept>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

__forceinline__ __device__ void getRect(
    const float2 p,
    int max_radius,
    uint2& rect_min,
    uint2& rect_max,
    dim3 grid)
{
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))};
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

#define CHECK_CUDA(A, debug) \
A; \
if (debug) { \
    auto ret = cudaDeviceSynchronize(); \
    if (ret != cudaSuccess) { \
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
        throw std::runtime_error(cudaGetErrorString(ret)); \
    } \
}

#endif
