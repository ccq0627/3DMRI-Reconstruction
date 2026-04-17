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

#ifndef CUDA_SLICE_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_SLICE_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    void preprocess(
        int P,
        const float* orig_points,
        const glm::vec3* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* cov3D_precomp,
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
        int* radii,
        float2* points_xy_image,
        float* depths,
        float* cov3Ds,
        float4* conic_opacity,
        const dim3 grid,
        uint32_t* tiles_touched,
        bool prefiltered);

    void render(
        const dim3 grid,
        dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W,
        int H,
        const float2* points_xy_image,
        const float4* conic_opacity,
        uint32_t* n_contrib,
        float* out_color);
}

#endif
