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

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

#include "forward.h"
#include "auxiliary.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Convert scale and quaternion to symmetric 3D covariance (upper triangle packed).
static __device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

template<int C>
__global__ void preprocessCUDA(
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
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	(void)prefiltered;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	if (slice_idx < 0 || slice_idx >= nVoxel_x)
		return;

	const float dVoxel_x = sVoxel_x / (float)nVoxel_x;
	const float dVoxel_y = sVoxel_y / (float)nVoxel_y;
	const float dVoxel_z = sVoxel_z / (float)nVoxel_z;

	const float depth_x = center_x - 0.5f * sVoxel_x + (slice_idx + 0.5f) * dVoxel_x;

	float3 p_orig = {
		orig_points[3 * idx + 0],
		orig_points[3 * idx + 1],
		orig_points[3 * idx + 2],
	};

	const float* cov3D = nullptr;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Σ packed as [Sxx, Sxy, Sxz, Syy, Syz, Szz]
	const float Sxx = cov3D[0];
	const float Sxy = cov3D[1];
	const float Sxz = cov3D[2];
	const float Syy = cov3D[3];
	const float Syz = cov3D[4];
	const float Szz = cov3D[5];

	const float eps_sxx = 1e-8f;
	if (Sxx <= eps_sxx)
		return;

	const float dx = depth_x - p_orig.x;
	const float inv_Sxx = 1.0f / Sxx;

	// Slice amplitude gate from fixing x at depth_x.
	const float slice_gate = expf(-0.5f * dx * dx * inv_Sxx);
	const float alpha_base = opacities[idx] * slice_gate;
	if (alpha_base < 1e-8f)
		return;

	// Conditional Gaussian in (y, z) | x=depth_x.
	const float mean_y = p_orig.y + Sxy * inv_Sxx * dx;
	const float mean_z = p_orig.z + Sxz * inv_Sxx * dx;

	float Cyy = Syy - Sxy * Sxy * inv_Sxx;
	float Cyz = Syz - Sxy * Sxz * inv_Sxx;
	float Czz = Szz - Sxz * Sxz * inv_Sxx;

	Cyy = fmaxf(Cyy, 1e-10f);
	Czz = fmaxf(Czz, 1e-10f);
	const float det_world = Cyy * Czz - Cyz * Cyz;
	if (det_world <= 1e-14f)
		return;

	// Convert world-space covariance in (y,z) to pixel-space covariance.
	const float inv_dy = 1.0f / dVoxel_y;
	const float inv_dz = 1.0f / dVoxel_z;
	const float Cyy_pix = Cyy * inv_dy * inv_dy;
	const float Cyz_pix = Cyz * inv_dy * inv_dz;
	const float Czz_pix = Czz * inv_dz * inv_dz;

	const float det_pix = Cyy_pix * Czz_pix - Cyz_pix * Cyz_pix;
	if (det_pix <= 1e-12f)
		return;

	const float det_inv = 1.0f / det_pix;
	const float3 conic = {
		//Czz_pix * det_inv,
		Cyy_pix * det_inv,
		-Cyz_pix * det_inv,
		//Cyy_pix * det_inv,
		Czz_pix * det_inv,
	};

	const float mid = 0.5f * (Cyy_pix + Czz_pix);
	const float eig_term = fmaxf(0.0f, mid * mid - det_pix);
	const float lambda1 = mid + sqrtf(eig_term);
	const float lambda2 = mid - sqrtf(eig_term);
	const float lambda_max = fmaxf(lambda1, lambda2);
	if (lambda_max <= 0.0f)
		return;

	int my_radius = (int)ceilf(3.0f * sqrtf(lambda_max));
	if (my_radius <= 0)
		return;

	// Pixel space: x axis is z, y axis is y.
	float2 point_image = {
		(mean_z - center_z + 0.5f * sVoxel_z) / dVoxel_z,
		(mean_y - center_y + 0.5f * sVoxel_y) / dVoxel_y,
	};

	// Fast reject if bounding circle is fully outside image.
	if (point_image.x + my_radius < 0.0f || point_image.y + my_radius < 0.0f ||
		point_image.x - my_radius > (float)nVoxel_z || point_image.y - my_radius > (float)nVoxel_y)
	{
		return;
	}

	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	depths[idx] = p_orig.x;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic_opacity[idx] = {conic.x, conic.y, conic.z, alpha_base};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W,
	int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_color)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	//float2 pixf = {(float)pix.x, (float)pix.y};
	float2 pixf = {(float)pix.x + 0.5f, (float)pix.y + 0.5f};
	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float4 con_o = collected_conic_opacity[j];

			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float alpha = con_o.w * expf(power);
			if (alpha < 1e-6f)
				continue;

			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += alpha;

			last_contributor = contributor;
		}
	}

	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch];
	}
}

void FORWARD::render(
	const dim3 grid,
	dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W,
	int H,
	const float2* means2D,
	const float4* conic_opacity,
	uint32_t* n_contrib,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W,
		H,
		means2D,
		conic_opacity,
		n_contrib,
		out_color);
}

void FORWARD::preprocess(
	int P,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
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
		means2D,
		depths,
		cov3Ds,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}
