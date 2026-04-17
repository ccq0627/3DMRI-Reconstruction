#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C
from .voxelization import GaussianVoxelizationSettings, voxelize_gaussians


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


class GaussianSliceRasterizationSettings(NamedTuple):
    slice_idx: int
    nVoxel_x: int
    nVoxel_y: int
    nVoxel_z: int
    sVoxel_x: float
    sVoxel_y: float
    sVoxel_z: float
    center_x: float
    center_y: float
    center_z: float
    scale_modifier: float
    prefiltered: bool
    debug: bool


def slice_rasterize_gaussians(
    means3D,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    slice_settings: GaussianSliceRasterizationSettings,
):
    return _SliceRasterizeGaussians.apply(
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        slice_settings,
    )


class _SliceRasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        slice_settings: GaussianSliceRasterizationSettings,
    ):
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            slice_settings.scale_modifier,
            cov3Ds_precomp,
            slice_settings.slice_idx,
            slice_settings.nVoxel_x,
            slice_settings.nVoxel_y,
            slice_settings.nVoxel_z,
            slice_settings.sVoxel_x,
            slice_settings.sVoxel_y,
            slice_settings.sVoxel_z,
            slice_settings.center_x,
            slice_settings.center_y,
            slice_settings.center_z,
            slice_settings.prefiltered,
            slice_settings.debug,
        )

        if slice_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = (
                    _C.slice_rasterize_gaussians(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_slice_fw.dump")
                print(
                    "\nAn error occured in slice forward. Please forward snapshot_slice_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = (
                _C.slice_rasterize_gaussians(*args)
            )

        ctx.slice_settings = slice_settings
        ctx.save_for_backward(means3D, opacities, scales, rotations, cov3Ds_precomp)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # Backward fallback uses existing differentiable voxelizer and extracts
        # the target depth slice to preserve training compatibility.
        slice_settings = ctx.slice_settings
        means3D, opacities, scales, rotations, cov3Ds_precomp = ctx.saved_tensors

        grad_means3D = None
        grad_opacities = None
        grad_scales = None
        grad_rotations = None
        grad_cov3Ds_precomp = None

        with torch.enable_grad():
            means3D_t = means3D.detach().requires_grad_(ctx.needs_input_grad[0])
            opacities_t = opacities.detach().requires_grad_(ctx.needs_input_grad[1])

            voxel_settings = GaussianVoxelizationSettings(
                scale_modifier=slice_settings.scale_modifier,
                nVoxel_x=slice_settings.nVoxel_x,
                nVoxel_y=slice_settings.nVoxel_y,
                nVoxel_z=slice_settings.nVoxel_z,
                sVoxel_x=slice_settings.sVoxel_x,
                sVoxel_y=slice_settings.sVoxel_y,
                sVoxel_z=slice_settings.sVoxel_z,
                center_x=slice_settings.center_x,
                center_y=slice_settings.center_y,
                center_z=slice_settings.center_z,
                prefiltered=slice_settings.prefiltered,
                debug=slice_settings.debug,
            )

            if cov3Ds_precomp.numel() > 0:
                cov3D_t = cov3Ds_precomp.detach().requires_grad_(ctx.needs_input_grad[4])
                fields, _ = voxelize_gaussians(
                    means3D_t,
                    opacities_t,
                    scales.detach(),
                    rotations.detach(),
                    cov3D_t,
                    voxel_settings,
                )
                pred_slice = fields[
                    slice_settings.slice_idx : slice_settings.slice_idx + 1, :, :
                ]
                grads = torch.autograd.grad(
                    outputs=pred_slice,
                    inputs=[means3D_t, opacities_t, cov3D_t],
                    grad_outputs=grad_out_color,
                    allow_unused=True,
                )
                grad_means3D, grad_opacities, grad_cov3Ds_precomp = grads
            else:
                scales_t = scales.detach().requires_grad_(ctx.needs_input_grad[2])
                rotations_t = rotations.detach().requires_grad_(ctx.needs_input_grad[3])
                fields, _ = voxelize_gaussians(
                    means3D_t,
                    opacities_t,
                    scales_t,
                    rotations_t,
                    cov3Ds_precomp.detach(),
                    voxel_settings,
                )
                pred_slice = fields[
                    slice_settings.slice_idx : slice_settings.slice_idx + 1, :, :
                ]
                grads = torch.autograd.grad(
                    outputs=pred_slice,
                    inputs=[means3D_t, opacities_t, scales_t, rotations_t],
                    grad_outputs=grad_out_color,
                    allow_unused=True,
                )
                grad_means3D, grad_opacities, grad_scales, grad_rotations = grads

        return (
            grad_means3D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


class GaussianSliceRasterizer(nn.Module):
    def __init__(self, slice_settings):
        super().__init__()
        self.slice_settings = slice_settings

    def forward(
        self,
        means3D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        slice_settings = self.slice_settings

        if not (0 <= int(slice_settings.slice_idx) < int(slice_settings.nVoxel_x)):
            raise ValueError(
                f"slice_idx {slice_settings.slice_idx} out of range [0, {slice_settings.nVoxel_x - 1}]"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        return slice_rasterize_gaussians(
            means3D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            slice_settings,
        )
