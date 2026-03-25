import os
import sys

import os.path as osp
import numpy as np
import argparse

sys.path.append("./")
from r2_gaussian.arguments import ParamGroup, ModelParams
from r2_gaussian.dataset import Scene

np.random.seed(0)

class InitParams_MRI(ParamGroup):
    def __init__(self, parser):
        self.n_points = 200_000
        self.density_thresh = 0.05
        self.density_rescale = 0.1
        super().__init__(parser, "Initialization Parameters")


def main(args, init_parser: InitParams_MRI, model_args: ModelParams):
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args)
    nii_cfg = scene.nii_cfg
    vol = scene.vol_gt_unsampled.cpu().numpy()
    # vol = np.load('MRIdata/under/pocs_recon.npy')

    save_path = args.output
    if not save_path:
        save_path = osp.join(data_path, f"acc_rate{model_args.accelerate_factor}", "Init_pointcloud" + ".npy")

    assert not osp.exists(
        save_path
    ), f"Initialization file {save_path} exists! Delete it first."
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    def init_pcd(
        args, save_path, nii_cfg, vol
    ):
        "Initialize spare points to create Gaussians."
        n_points = args.n_points
        density_mask = vol > args.density_thresh
        valid_indices = np.argwhere(density_mask)
        
        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        offOrigin = np.array(nii_cfg["offOrigin"])
        dVoxel = np.array(nii_cfg["dVoxel"])
        sVoxel = np.array(nii_cfg["sVoxel"])

        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + dVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]

        sampled_densities = sampled_densities * args.density_rescale
        out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
        np.save(save_path, out)
        print(f"Initialization saved in {save_path}.")

    init_pcd(
        args=init_parser, 
        save_path=save_path,
        nii_cfg=nii_cfg,
        vol=vol,
    )



if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    init_parser = InitParams_MRI(parser)
    parser.add_argument("--data", type=str, help="Path to data.",default="MRIdata")
    parser.add_argument("--output", type=str, help="Output path", default=None)
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args), lp.extract(args))
