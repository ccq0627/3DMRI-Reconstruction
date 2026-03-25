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
import logging
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
from r2_gaussian.utils.log_utils import prepare_output_and_logger, setup_experiment_folder
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, L2_loss, l1_loss_image
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice


def install_tqdm_write_logger(log_path="log.txt", level=logging.INFO):
    """Patch tqdm.write so messages are also appended to a log file."""
    logger = logging.getLogger("tqdm_write_logger")
    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(osp.dirname(log_path) or ".", exist_ok=True)

    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if getattr(tqdm, "_write_logger_installed", False):
        return

    original_write = tqdm.write

    def write_and_log(cls, s, file=None, end="\n", nolock=False):
        original_write(s, file=file, end=end, nolock=nolock)
        logger.log(level, str(s).rstrip("\n"))

    tqdm.write = classmethod(write_and_log)
    tqdm._write_logger_installed = True

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
    gaussians.training_setup(opt)

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
        pred_vol_image = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(pred_vol_kspace * mask), norm='ortho')
        )
        gt_vol_image = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(gt_vol_kspace * mask), norm='ortho')
        )

        dc_loss_image = l1_loss_image(pred_vol_image, gt_vol_image)
        loss["dc_loss_image"] = dc_loss_image
        loss["total"] += 0 * loss["dc_loss_image"]
        
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
            """wait to writting"""
            # Adaptive control
            gaussians.add_densification_stats()
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold, # 0.2
                        bbox,
                    )
            
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
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                queryfunc,
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

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

        # if iteration == testing_iterations[-1]:

        # tv_writer = None (default)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, pts: {scene.gaussians.get_xyz.shape[0]:5d}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
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

    args.model_path = setup_experiment_folder(op, lp)
    # set up log path
    log_path = osp.join(args.model_path, "log.txt")
    install_tqdm_write_logger(log_path)
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
    
