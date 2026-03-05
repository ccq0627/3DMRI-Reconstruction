"""
write for test
"""

import torch
import math 
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


# 渲染初始化的高斯
parser = ArgumentParser(description="R2 Gaussian")
mp = ModelParams(parser)
pp = PipelineParams(parser)
args = parser.parse_args()
args_dict = vars(args)
args_dict["source_path"] = 'data/naf_dataset/head_50.pickle'
args_dict["model_path"] = 'data/naf_dataset/output_2026_3_5'

dataset = mp.extract(args)
pipe = pp.extract(args)

scene = Scene(dataset,shuffle=False)

train_cameras = scene.getTrainCameras()

gaussians = GaussianModel(None)
initialize_gaussian(gaussians, dataset, None)
scene.gaussians = gaussians

## 取第一个相机
viewpoint_cam = train_cameras[0]

mean2D = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0.0
mode = viewpoint_cam.mode
tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
raster_setting = GaussianRasterizationSettings(
    image_height=int(viewpoint_cam.image_height),
    image_width=int(viewpoint_cam.image_width),
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    scale_modifier=1.0,
    viewmatrix=viewpoint_cam.world_view_transform,
    projmatrix=viewpoint_cam.full_proj_transform,
    campos=viewpoint_cam.camera_center,
    prefiltered=False,
    mode=mode,
    debug=pipe.debug,
)
rasterizer = GaussianRasterizer(raster_settings=raster_setting)

mean3D = gaussians.get_xyz
density = gaussians.get_density
scales = gaussians.get_scaling
rotations = gaussians.get_rotation

rendered_image, radii = rasterizer(
    means3D=mean3D,
    means2D=mean2D,
    opacities=density,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
)
if True:
    img = rendered_image.detach().cpu().clone().numpy().transpose(1,2,0)
    plt.figure(figsize=(10,20))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Rendering image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(viewpoint_cam.original_image.cpu().numpy().transpose(1,2,0))
    plt.title("Original image")
    plt.axis("off")

    plt.show()


print(1)


