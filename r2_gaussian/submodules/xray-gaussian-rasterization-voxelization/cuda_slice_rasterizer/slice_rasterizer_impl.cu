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

#include "slice_rasterizer_impl.h"
#include <algorithm>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iostream>
#include <numeric>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"

static uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

static __global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    if (radii[idx] > 0)
    {
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;
        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths[idx]);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

static __global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

CudaSliceRasterizer::GeometryState CudaSliceRasterizer::GeometryState::fromChunk(
    char*& chunk,
    size_t P)
{
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaSliceRasterizer::ImageState CudaSliceRasterizer::ImageState::fromChunk(
    char*& chunk,
    size_t N)
{
    ImageState img;
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);
    return img;
}

CudaSliceRasterizer::BinningState CudaSliceRasterizer::BinningState::fromChunk(
    char*& chunk,
    size_t P)
{
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        binning.sorting_size,
        binning.point_list_keys_unsorted,
        binning.point_list_keys,
        binning.point_list_unsorted,
        binning.point_list,
        P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

int CudaSliceRasterizer::SliceRasterizer::forward(
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
    int* radii,
    bool debug)
{
    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if (radii == nullptr)
    {
        radii = geomState.internal_radii;
    }

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

    CHECK_CUDA(FORWARD::preprocess(
                   P,
                   means3D,
                   (glm::vec3*)scales,
                   scale_modifier,
                   (glm::vec4*)rotations,
                   opacities,
                   cov3D_precomp,
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
                   radii,
                   geomState.means2D,
                   geomState.depths,
                   geomState.cov3D,
                   geomState.conic_opacity,
                   tile_grid,
                   geomState.tiles_touched,
                   prefiltered),
               debug)

    int num_rendered = 0;
    if (P > 0)
    {
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                       geomState.scanning_space,
                       geomState.scan_size,
                       geomState.tiles_touched,
                       geomState.point_offsets,
                       P),
                   debug)

        CHECK_CUDA(cudaMemcpy(
                       &num_rendered,
                       geomState.point_offsets + P - 1,
                       sizeof(int),
                       cudaMemcpyDeviceToHost),
                   debug)
    }

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug)

    if (num_rendered > 0)
    {
        size_t binning_chunk_size = required<BinningState>(num_rendered);
        char* binning_chunkptr = binningBuffer(binning_chunk_size);
        BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

        duplicateWithKeys<<<(P + 255) / 256, 256>>>(
            P,
            geomState.means2D,
            geomState.depths,
            geomState.point_offsets,
            binningState.point_list_keys_unsorted,
            binningState.point_list_unsorted,
            radii,
            tile_grid);
        CHECK_CUDA(, debug)

        int bit = getHigherMsb(tile_grid.x * tile_grid.y);

        CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                       binningState.list_sorting_space,
                       binningState.sorting_size,
                       binningState.point_list_keys_unsorted,
                       binningState.point_list_keys,
                       binningState.point_list_unsorted,
                       binningState.point_list,
                       num_rendered,
                       0,
                       32 + bit),
                   debug)

        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
        CHECK_CUDA(, debug)

        CHECK_CUDA(FORWARD::render(
                       tile_grid,
                       block,
                       imgState.ranges,
                       binningState.point_list,
                       width,
                       height,
                       geomState.means2D,
                       geomState.conic_opacity,
                       imgState.n_contrib,
                       out_slice),
                   debug)
    }

    return num_rendered;
}
