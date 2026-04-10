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

import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import torch.nn.functional as F

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, get_mask
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger, setup_experiment_folder, prepare_tqdm_write_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, l1_loss_image, edge_loss_fn
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
from metric_MRI import evaluate_slices, THRESHOLD


# 原始数据： k_space (ifft)->  MRI (initialize gs)-> TV loss (query)-> 
# pred MRI (fft)-> pred kspace -> compute loss
def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset)
    gt_vol_kspace = scene.vol_gt_kspace  # device = cuda  欠采样点的kspace
    mask = scene.mask

    # Set up some parameters
    nii_cfg = scene.nii_cfg
    bbox = scene.bbox
    volume_to_world = max(nii_cfg["sVoxel"])
    # opt.max_scale = 0.15 (dafult)
    # max_scale = 0.3
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    # opt.densify_scale_threshold = 0.1 (percent of volume size)
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        # default scale_min=0.0005, scale_max=0.5
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world  # [0.001, 1.0]
    queryfunc = lambda x: query(
        x,
        nii_cfg["offOrigin"],
        nii_cfg["nVoxel"],
        nii_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)  # Set up optimizer and scheduler
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = False
    if opt.lambda_tv is not None:
        use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
    if opt.use_image_loss:
        print("Use image domain loss")

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)

    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # query volume
        pred_vol = queryfunc(gaussians)["vol"]  # 图像域 device:cuda
        # fft获得kspace
        if not pred_vol.is_complex():
            pred_vol_complex = torch.complex(pred_vol, torch.zeros_like(pred_vol))
        else:
            pred_vol_complex = pred_vol
        pred_vol_kspace = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(pred_vol_complex), norm='ortho')
        )
        
        # # Compute loss
        # 直接使用体积compute loss
        loss = {"total": 0.0}
        pred_sampled = pred_vol_kspace[mask.bool()]
        gt_sampled = gt_vol_kspace[mask.bool()] # gt_vol_kspace 必须只有实测的那些点有值
        
        dc_loss = l1_loss(pred_sampled, gt_sampled)

        loss["dc_loss"] = dc_loss
        
        loss["total"] += loss["dc_loss"]

        # image domain loss

        # IFFT compute pred image and gt image
        if opt.use_image_loss:
            pred_vol_image = torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(pred_vol_kspace * mask), norm='ortho')
            )
            gt_vol_image = torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(gt_vol_kspace * mask), norm='ortho')
            )

            # dc_loss_image = l1_loss_image(pred_vol_image, gt_vol_image)
            # loss["dc_loss_image"] = dc_loss_image
            # loss["total"] += loss["dc_loss_image"]
            
            edge_loss = edge_loss_fn(pred_vol_image, gt_vol_image)
            loss["edge_loss"] = edge_loss
            loss["total"] += opt.lambda_edge * loss["edge_loss"]

            
            # 2D SSIM on each slice (first dim is depth): [D, H, W] -> [D, 1, H, W]
            pred_slices_2d = torch.abs(pred_vol_image).unsqueeze(1)
            gt_slices_2d = torch.abs(gt_vol_image).unsqueeze(1)
            ssim_loss = 1 - ssim(pred_slices_2d, gt_slices_2d)
            loss["ssim_loss"] = ssim_loss
            
            loss["total"] += opt.lambda_dssim * loss["ssim_loss"]

        # 3D TV loss
        if use_tv:
            loss_tv = tv_3d_loss(pred_vol, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats()
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,  # 决定是否致密化
                        opt.density_min_threshold,  # Prune贡献小的高斯
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold, # 0.2， 决定克隆还是分裂
                        bbox,
                        opt.use_las,
                    )
            if False:
                print("======================================================================")
                print(f"[ITER {iteration}] Number of Gaussians: {gaussians.get_xyz.shape[0]}")
                print(f"[ITER {iteration}] Loss: {loss['total'].item():.7f}, DC Loss: {loss['dc_loss'].item():.7f}")
                print(f"[ITER {iteration}] Gaussians xyz: {gaussians.get_xyz}")
                print(f"[ITER {iteration}] Gaussians xyz grad: {gaussians._xyz.grad}")
                
                print(f"[ITER {iteration}] Gaussians density: {gaussians.get_density}")
                print(f"[ITER {iteration}] Gaussians original density: {gaussians._density}")
                print(f"[ITER {iteration}] Gaussians density grad: {gaussians._density.grad}")
                
                print(f"[ITER {iteration}] Gaussians scale: {gaussians.get_scaling}")
                print(f"[ITER {iteration}] Gaussians original scale: {gaussians._scaling}")
                print(f"[ITER {iteration}] Gaussians scale grad: {gaussians._scaling.grad}")

            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                tqdm.write(f"[ITER {iteration}] Computing Loss: {loss['total'].item():.7f}")
                scene.save(iteration, queryfunc)

            # Metrics
            if iteration == opt.iterations:
                if dataset.eval:
                    point_cloud_path = osp.join(
                        scene.model_path, "point_cloud/iteration_{}".format(iteration)
                    )
                    vol_gt_path = osp.join(point_cloud_path, "vol_gt.npy")
                    vol_pred_path = osp.join(point_cloud_path, "vol_pred.npy")
                    output_path = osp.join(scene.model_path,"eval.yaml")
                    if osp.exists(vol_gt_path) and osp.exists(vol_pred_path):
                        vol_gt_eval = torch.from_numpy(np.load(vol_gt_path)).float()
                        vol_pred_eval = torch.from_numpy(np.load(vol_pred_path)).float()
                        pixel_max = float(vol_gt_eval.max().item())
                        eval_result = evaluate_slices(
                            vol_gt_eval,
                            vol_pred_eval,
                            pixel_max=pixel_max,
                            min_tissue=THRESHOLD,
                        )
                        eval_result["gt_path"] = vol_gt_path
                        eval_result["pred_path"] = vol_pred_path
                        eval_result["pixel_max"] = pixel_max
                        with open(output_path, "w", encoding="utf-8") as f:
                            yaml.safe_dump(
                                eval_result, f, sort_keys=False, allow_unicode=False
                            )
                        tqdm.write(
                            f"[ITER {iteration}] Saved slice-wise eval to {output_path}. "
                            f"PSNR={eval_result['mean']['psnr']:.4f}, SSIM={eval_result['mean']['ssim']:.4f}, "
                            f"valid={eval_result['num_valid_slices']}/{eval_result['num_slices']}"
                        )
                    else:
                        tqdm.write(
                            f"[ITER {iteration}] Skip slice-wise eval: missing {vol_gt_path} or {vol_pred_path}."
                        )

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt_" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 5 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.7f}",
                        "pts": f"{gaussians.get_density.shape[0]}",
                    }
                )
                progress_bar.update(5)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                iteration,
                testing_iterations,
                scene,
                queryfunc,
            )


def training_report(
    iteration,
    testing_iterations,
    scene: Scene,
    queryFunc,
):
    # Add training statistics

    if iteration in testing_iterations:
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt  # device = cuda
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, pts: {scene.gaussians.get_xyz.shape[0]:5d}"
        )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1,100,200,300,400, 500, 1000,1500,2000,2500,3000,5000,10000,15000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1,100,200,300,400, 500, 1000,1500,2000,2500,3000,5000,10000,15000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default='config/config_MRI.yaml', help="Path of config")  # debug config file
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    args.model_path = setup_experiment_folder(args)
    # set up log path
    log_path = osp.join(args.model_path, "log.txt")
    prepare_tqdm_write_logger(log_path)
    # Set up logging writer
    # Set up output folder(if None) and return SummaryWriter
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
    
