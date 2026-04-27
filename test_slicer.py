import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

from argparse import ArgumentParser


from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, query, slice_rasterize, initialize_gaussian
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, edge_loss_fn

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = ArgumentParser()
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)

args = parser.parse_args(sys.argv[1:])

cfg = load_config("config/config_MRI.yaml")
vars(args).update(cfg) 
args.iterations = 100

dataset = lp.extract(args)
opt = op.extract(args)
pipe = pp.extract(args)

scene = Scene(dataset)
gt_vol_kspace = scene.vol_gt_kspace
mask = scene.mask
nii_cfg = scene.nii_cfg
volume_to_world = max(nii_cfg["sVoxel"])

scale_bound = None
if dataset.scale_min > 0 and dataset.scale_max > 0:
    scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

queryfunc = lambda x: query(
    x,
    nii_cfg["offOrigin"],
    nii_cfg["nVoxel"],
    nii_cfg["sVoxel"],
    pipe,
)

gaussians = GaussianModel(scale_bound)
initialize_gaussian(gaussians, dataset, None)
gaussians.load_ply("MRIdata/outputs/exp_04-14_19-11_iter3000_8_wi_sigma30/point_cloud/iteration_3000/point_cloud.pickle")
scene.gaussians = gaussians
# gaussians.training_setup(opt)


depth_dim = int(nii_cfg["nVoxel"][0])
slice_idx = random.randint(0, depth_dim - 1)
slice_result = slice_rasterize(
    gaussians,
    128,
    nii_cfg["offOrigin"],
    nii_cfg["nVoxel"],
    nii_cfg["sVoxel"],
    pipe,
)
pred_slice = slice_result["render"]

plt.figure()
plt.imshow(pred_slice.clone().detach().permute(1,2,0).cpu().numpy(), cmap="gray")
plt.savefig("slice_rasterize_result.png")


# use_tv = opt.lambda_tv is not None and opt.lambda_tv > 0
# checkpoints = [1] + list(range(10, 101, 10))
# rows = []

# for iteration in range(1, opt.iterations + 1):
#     gaussians.update_learning_rate(iteration)

#     pred_vol = queryfunc(gaussians)["vol"]
#     if not pred_vol.is_complex():
#         pred_vol_complex = torch.complex(pred_vol, torch.zeros_like(pred_vol))
#     else:
#         pred_vol_complex = pred_vol

#     pred_vol_kspace = torch.fft.fftshift(
#         torch.fft.fftn(torch.fft.ifftshift(pred_vol_complex), norm="ortho")
#     )

#     loss = {}
#     pred_sampled = pred_vol_kspace[mask.bool()]
#     gt_sampled = gt_vol_kspace[mask.bool()]
#     loss["dc_loss"] = l1_loss(pred_sampled, gt_sampled)
#     total = loss["dc_loss"]

#     if opt.use_image_loss:
#         pred_vol_image = torch.fft.fftshift(
#             torch.fft.ifftn(torch.fft.ifftshift(pred_vol_kspace * mask), norm="ortho")
#         )
#         gt_vol_image = torch.fft.fftshift(
#             torch.fft.ifftn(torch.fft.ifftshift(gt_vol_kspace * mask), norm="ortho")
#         )
#         loss["edge_loss"] = edge_loss_fn(pred_vol_image, gt_vol_image)
#         total = total + opt.lambda_edge * loss["edge_loss"]

#         pred_slices_2d = torch.abs(pred_vol_image).unsqueeze(1)
#         gt_slices_2d = torch.abs(gt_vol_image).unsqueeze(1)
#         loss["ssim_loss"] = 1 - ssim(pred_slices_2d, gt_slices_2d)
#         total = total + opt.lambda_dssim * loss["ssim_loss"]

#     if use_tv:
#         loss["tv"] = tv_3d_loss(pred_vol, reduction="mean")
#         total = total + opt.lambda_tv * loss["tv"]

#     if opt.use_slice_rasterizer and opt.lambda_slice > 0:
#         depth_dim = int(nii_cfg["nVoxel"][0])
#         slice_idx = random.randint(0, depth_dim - 1)
#         slice_result = slice_rasterize(
#             gaussians,
#             slice_idx,
#             nii_cfg["offOrigin"],
#             nii_cfg["nVoxel"],
#             nii_cfg["sVoxel"],
#             pipe,
#         )
#         pred_slice = slice_result["render"]
#         gt_slice = scene.vol_gt[slice_idx : slice_idx + 1, :, :]
#         loss["slice_loss"] = l1_loss(pred_slice, gt_slice)
#         total = total + opt.lambda_slice * loss["slice_loss"]
#     else:
#         loss["slice_loss"] = torch.zeros((), device=total.device, dtype=total.dtype)

#     total.backward()

#     if iteration < opt.iterations:
#         gaussians.optimizer.step()
#         gaussians.optimizer.zero_grad(set_to_none=True)

#     if iteration in checkpoints:
#         rows.append((
#             iteration,
#             float(total.item()),
#             float(loss["dc_loss"].item()),
#             float(loss["slice_loss"].item()),
#         ))

# with open(r"_tmp_loss_trend_100.csv", "w", encoding="utf-8") as f:
#     f.write("iter,total_loss,dc_loss,slice_loss\n")
#     for r in rows:
#         f.write(f"{r[0]},{r[1]:.7f},{r[2]:.7f},{r[3]:.7f}\n")

# print("done")
