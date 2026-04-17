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

#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SliceRasterizeGaussiansCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const int slice_idx,
    const int nVoxel_x,
    const int nVoxel_y,
    const int nVoxel_z,
    const float sVoxel_x,
    const float sVoxel_y,
    const float sVoxel_z,
    const float center_x,
    const float center_y,
    const float center_z,
    const bool prefiltered,
    const bool debug);
