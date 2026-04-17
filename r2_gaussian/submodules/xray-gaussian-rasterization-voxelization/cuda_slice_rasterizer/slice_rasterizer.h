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

#ifndef CUDA_SLICE_RASTERIZER_H_INCLUDED
#define CUDA_SLICE_RASTERIZER_H_INCLUDED

#include <functional>

namespace CudaSliceRasterizer
{
class SliceRasterizer
{
public:
    static int forward(
        std::function<char*(size_t)> geometryBuffer,
        std::function<char*(size_t)> binningBuffer,
        std::function<char*(size_t)> imageBuffer,
        const int P,
        const int width,
        const int height,
        const int nVoxel_x,
        const int nVoxel_y,
        const int nVoxel_z,
        const float sVoxel_x,
        const float sVoxel_y,
        const float sVoxel_z,
        const float center_x,
        const float center_y,
        const float center_z,
        const int slice_idx,
        const float* means3D,
        const float* opacities,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const bool prefiltered,
        float* out_slice,
        int* radii = nullptr,
        bool debug = false);
};

} // namespace CudaSliceRasterizer

#endif
