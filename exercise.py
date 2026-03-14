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
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
from r2_gaussian.gaussian import query


# 渲染初始化的高斯
parser = ArgumentParser(description="R2 Gaussian")
mp = ModelParams(parser)
pp = PipelineParams(parser)
args = parser.parse_args()
args_dict = vars(args)
args_dict["source_path"] = 'MRIdata'
args_dict["model_path"] = 'data/naf_dataset/output_2026_3_5'

dataset = mp.extract(args)
pipe = pp.extract(args)

scene = Scene(dataset)

scale_bound = [0.001, 1.0]

gaussians = GaussianModel(scale_bound)
# initialize_gaussian(gaussians, dataset, None)
scene.gaussians = gaussians
ply_path = 'MRIdata/output_2026_3_6/point_cloud/iteration_1000/point_cloud.pickle'
gaussians.load_ply(ply_path)

print(gaussians.get_scaling)

queryfunc = lambda x: query(
        x,
        [0,0,0],
        [32,32,32],
        [1.4914,2,2],
        pipe,
    )

gt_vol = queryfunc(gaussians)["vol"]


import pyvista as pv
plotter = pv.Plotter(window_size=(800,800))
plotter.subplot(0, 0)
plotter.add_volume(gt_vol.cpu().detach().numpy(), cmap="viridis")
plotter.show()
print(1)


