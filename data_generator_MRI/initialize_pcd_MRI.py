import os
import sys

import os.path as osp
import numpy as np
import argparse

sys.path.append("./")
from r2_gaussian.arguments import ParamGroup


np.random.seed(0)

class InitParams_MRI(ParamGroup):
    def __init__(self, parser):
        self.n_points = 60_000
        self.density_thresh = 0.5
        self.density_rescale = 0.15
        super().__init__(parser, "Initialization Parameters")


def init_pcd(
        args, save_path, source_path
):
    "Initialize spare points to create Gaussians."
    n_points = args.n_points
    vol = np.load(source_path)
    density_mask = vol > args.density_thresh
    valid_indices = np.argwhere(density_mask)
    
    offOrigin = np.array([0, 0, 0])
    sVoxel = np.array([2.0, 2.0, 2.0])
    nVoxel = np.array(vol.shape)
    dVoxel = sVoxel / nVoxel

    scanner_cfg = {
        "offOrigin": offOrigin.tolist(),
        "dVoxel": dVoxel.tolist(),
        "nVoxel": nVoxel.tolist(),
        "sVoxel": sVoxel.tolist()
    }
    scene_scale = 2 / max(scanner_cfg["sVoxel"])

    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "offOrigin",
    ]:
        scanner_cfg[key_to_scale] = (
            np.array(scanner_cfg[key_to_scale]) * scene_scale
        ).tolist()

    sampled_indices = valid_indices[
        np.random.choice(len(valid_indices), n_points, replace=False)
    ]
    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]

    sampled_densities = sampled_densities * args.density_rescale
    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")



def main(args, init_parser: InitParams_MRI):
    dataset_path = args.data
    save_path = args.output
    if not save_path:
        dir_path = osp.dirname(dataset_path)
        save_path = osp.join(dir_path, "Init_pointcloud" + ".npy")
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    init_pcd(
        args=init_parser, 
        save_path=save_path,
        source_path=dataset_path,
    )



if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    init_parser = InitParams_MRI(parser)
    parser.add_argument("--data", type=str, help="Path to preprocess dataset.",default="MRIdata/vol_gt.npy")
    parser.add_argument("--output", type=str, help="Output path", default=None)
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args))
