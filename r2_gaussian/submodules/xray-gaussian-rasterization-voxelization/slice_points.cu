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

#include <cuda_runtime_api.h>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

#include "cuda_slice_rasterizer/config.h"
#include "cuda_slice_rasterizer/slice_rasterizer.h"
#include "utility.h"

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
    const bool debug)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3)
    {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);
    const int H = nVoxel_y;
    const int W = nVoxel_z;

    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_slice = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if (P != 0)
    {
        rendered = CudaSliceRasterizer::SliceRasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P,
            W,
            H,
            nVoxel_x,
            nVoxel_y,
            nVoxel_z,
            sVoxel_x,
            sVoxel_y,
            sVoxel_z,
            center_x,
            center_y,
            center_z,
            slice_idx,
            means3D.contiguous().data<float>(),
            opacity.contiguous().data<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(),
            prefiltered,
            out_slice.contiguous().data<float>(),
            radii.contiguous().data<int>(),
            debug);
    }

    return std::make_tuple(rendered, out_slice, radii, geomBuffer, binningBuffer, imgBuffer);
}
