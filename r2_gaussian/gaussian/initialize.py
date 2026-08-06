import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.utils.graphics_utils import fetchPly
from r2_gaussian.utils.system_utils import searchForMaxIteration


def initialize_gaussian(
    gaussians: GaussianModel, 
    args: ModelParams,
    vol: np.ndarray,
    nii_cfg: dict, 
    loaded_iter=None,
    n_points = 200_000,
    density_thresh = 0.05,
    density_rescale = 0.1,
):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )
        assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
        gaussians.load_ply(ply_path)
        print("Loading trained model at iteration {}".format(loaded_iter))
    else:
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                )
            # add MRI format
            elif osp.exists(osp.join(args.source_path, "nii_data.json")):
                ply_path = osp.join(
                    args.source_path, "Init_pointcloud.npy"
                )
            else:
                raise ValueError("Could not recognize scene type!")
        else:
            ply_path = args.ply_path

        if not osp.exists(ply_path):
            print(f"Cannot find {ply_path} for initialization. Generating a new point cloud with ifft result.")
            density_mask = vol > density_thresh
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
    
            sampled_densities = sampled_densities * density_rescale

            gaussians.create_from_pcd(sampled_positions, sampled_densities, 1.0)

        
        else:
            print(f"Initialize Gaussians with {osp.basename(ply_path)} in {osp.dirname(ply_path)}.")
            ply_type = ply_path.split(".")[-1]
            if ply_type == "npy":
                point_cloud = np.load(ply_path)
                xyz = point_cloud[:, :3]
                density = point_cloud[:, 3:4]
            elif ply_type == ".ply":
                point_cloud = fetchPly(ply_path)
                xyz = np.asarray(point_cloud.points)
                density = np.asarray(point_cloud.colors[:, :1])

            gaussians.create_from_pcd(xyz, density, 1.0)

    return loaded_iter
