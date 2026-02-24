import os
import os.path as osp
import sys
import argparse
import glob
import numpy as np
from tqdm import trange
import tigre.algorithms as algs
import cv2
import random
import json
import SimpleITK as sitk

random.seed(0)

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre


def main(args):
    input_data_path = args.data
    proj_subsample = args.proj_subsample
    proj_rescale = args.proj_rescale
    object_scale = args.object_scale

    # Read configuration
    config_file_path = osp.join(input_data_path, "config.txt")
    if not osp.exists(config_file_path):
        print(f"Warning: config.txt not found in {input_data_path}. Please ensure geometry parameters are provided.")
        # Define defaults or exit if critical
        n_proj = 0
        angle_interval = 0
        angle_start = 0
        angle_last = 0
        DSD = 0
        DSO = 0
        dDetector = 0
    else:
        with open(config_file_path, "r") as f:
            for config_line in f.readlines():
                if "NumberImages" in config_line:
                    n_proj = int(config_line.split("=")[-1])
                elif "AngleInterval" in config_line:
                    angle_interval = float(config_line.split("=")[-1])
                elif "AngleFirst" in config_line:
                    angle_start = float(config_line.split("=")[-1])
                elif "AngleLast" in config_line:
                    angle_last = float(config_line.split("=")[-1])
                elif "DistanceSourceDetector" in config_line:
                    DSD = float(config_line.split("=")[-1]) / 1000 * object_scale
                elif "DistanceSourceOrigin" in config_line:
                    DSO = float(config_line.split("=")[-1]) / 1000 * object_scale
                elif "PixelSize" in config_line and "PixelSizeUnit" not in config_line:
                    dDetector = (
                        float(config_line.split("=")[-1])
                        * proj_subsample
                        / 1000
                        * object_scale
                    )
    
    # Locate MHD file
    mhd_files = glob.glob(osp.join(input_data_path, "*.mhd"))
    if len(mhd_files) == 0:
        raise FileNotFoundError(f"No .mhd files found in {input_data_path}")
    mhd_path = mhd_files[0]
    print(f"Loading data from {mhd_path}...")
    
    # Read MHD
    itk_img = sitk.ReadImage(mhd_path)
    # Shape is usually (N_proj, Height, Width) or (N_proj, Width, Height) depending on file
    # We assume standard order (z, y, x) from SimpleITK which maps to (N, H, W)
    projections_stack = sitk.GetArrayFromImage(itk_img) 
    
    n_proj_loaded = projections_stack.shape[0]
    if n_proj != n_proj_loaded:
        print(f"Warning: Config says {n_proj} images, but loaded {n_proj_loaded}. Using loaded count.")
        n_proj = n_proj_loaded

    # Calculate angles
    # If angle info is missing, we might need to assume 360 degrees or read from somewhere else
    if angle_interval == 0:
         # Try to infer if AngleLast and AngleFirst are present
         if angle_last != angle_start:
             angle_interval = (angle_last - angle_start) / (n_proj - 1)
         else:
             print("Warning: Angle interval not found. Assuming 360 degrees rotation.")
             angle_start = 0
             angle_last = 360
             angle_interval = 360.0 / n_proj

    angles = np.linspace(angle_start, angle_last, n_proj)
    angles = angles / 180.0 * np.pi  # Convert to radians

    # Prepare output directories
    output_path = args.output
    all_save_path = osp.join(output_path, "proj_all")
    train_save_path = osp.join(output_path, "proj_train")
    test_save_path = osp.join(output_path, "proj_test")
    os.makedirs(all_save_path, exist_ok=True)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    projection_train_list = []
    projection_test_list = []
    
    # Train/Test split
    # Ensure n_train is valid
    if args.n_train > n_proj:
        args.n_train = int(n_proj * 0.8)
    if args.n_test > n_proj - args.n_train:
        args.n_test = n_proj - args.n_train

    train_ids = np.linspace(0, n_proj - 1, args.n_train).astype(int)
    possible_test_ids = np.setdiff1d(np.arange(n_proj), train_ids).tolist()
    if len(possible_test_ids) < args.n_test:
        test_ids = sorted(possible_test_ids)
    else:
        test_ids = sorted(random.sample(possible_test_ids, args.n_test))

    # Process and Save
    for i_proj in trange(n_proj, desc="Processing projections"):
        proj = projections_stack[i_proj]
        
        # Normalize/Rescale
        # Assuming input is raw density or similar, rescale to [0, 1] range if needed
        # Or match the logic: value / rescale * object_scale
        proj = proj / proj_rescale * object_scale
        proj = proj.astype(np.float32)
        proj[proj < 0] = 0
        
        # Optional: Dataset specific shifts (Commented out as it might not apply generally)
        # proj_new = np.zeros_like(proj)
        # proj_new[:-5] = proj[5:]
        # proj = proj_new

        # Subsample / Resize
        if proj_subsample != 1.0:
            h_ori, w_ori = proj.shape
            h_new, w_new = int(h_ori / proj_subsample), int(w_ori / proj_subsample)
            proj = cv2.resize(proj, (w_new, h_new))
            
            # Crop to ensure even dimensions or specific requirements if needed
            # Here we keep original logic: crop to rectangle (centered) if needed? 
            # The original script does: crop max dimension to match min dimension?
            # Actually original code:
            # if dim_x > dim_y: crop x ...
            # This makes it square. Let's keep it optional or strictly follow original if we want square projections.
            # Assuming we want to maintain the pipeline compatibility:
            dim_x, dim_y = proj.shape
            if dim_x > dim_y:
                dim_offset = int((dim_x - dim_y) / 2)
                proj = proj[dim_offset:-dim_offset, :]
            elif dim_x < dim_y:
                dim_offset = int((dim_y - dim_x) / 2)
                proj = proj[:, dim_offset:-dim_offset]

        # Save
        proj_save_name = f"proj_{i_proj:04d}"
        
        if i_proj in train_ids:
            projection_train_list.append(
                {
                    "file_path": osp.join(
                        osp.basename(train_save_path), proj_save_name + ".npy"
                    ),
                    "angle": angles[i_proj],
                }
            )
            np.save(osp.join(train_save_path, proj_save_name + ".npy"), proj)
        elif i_proj in test_ids:
            projection_test_list.append(
                {
                    "file_path": osp.join(
                        osp.basename(test_save_path), proj_save_name + ".npy"
                    ),
                    "angle": angles[i_proj],
                }
            )
            np.save(osp.join(test_save_path, proj_save_name + ".npy"), proj)
            
        np.save(osp.join(all_save_path, proj_save_name + ".npy"), proj)

    # Reconstruction Config
    # We need to know the shape of the processed projection to set detector size
    sample_proj = np.load(osp.join(all_save_path, f"proj_{train_ids[0]:04d}.npy"))
    nDetector = [sample_proj.shape[0], sample_proj.shape[1]]
    sDetector = np.array(nDetector) * np.array(dDetector)
    
    nVoxel = args.nVoxel
    sVoxel = args.sVoxel
    offOrigin = args.offOrigin
    
    bbox = np.array(
        [
            np.array(offOrigin) - np.array(sVoxel) / 2,
            np.array(offOrigin) + np.array(sVoxel) / 2,
        ]
    ).tolist()
    
    scanner_cfg = {
        "mode": "cone",
        "DSD": DSD,
        "DSO": DSO,
        "nDetector": nDetector,
        "sDetector": sDetector.tolist(),
        "nVoxel": nVoxel,
        "sVoxel": sVoxel,
        "offOrigin": offOrigin,
        "offDetector": args.offDetector,
        "accuracy": args.accuracy,
        "totalAngle": angle_last - angle_start,
        "startAngle": angle_start,
        "noise": True, # Real data implies noise is inherent, this might be a flag for reconstruction or synthetic gen
        "filter": None,
    }

    # FDK Reconstruction
    ct_gt_save_path = osp.join(output_path, "vol_gt.npy")
    if not osp.exists(ct_gt_save_path):
        print("Reconstructing with FDK...")
        
        # Load all projections for reconstruction
        # Use simple sliceing if memory allows, or load incrementally if needed. 
        # Original script loads all.
        projs_for_recon = []
        # We assume we want to use all projections or a subset?
        # Original: uses all projections in 'proj_all'
        
        # To avoid sorting issues, we iterate by index since we named them with IDs
        proj_stack_recon = []
        for i in range(n_proj):
             p = np.load(osp.join(all_save_path, f"proj_{i:04d}.npy"))
             proj_stack_recon.append(p)
        
        projs = np.stack(proj_stack_recon, axis=0)
        
        geo = get_geometry_tigre(scanner_cfg)
        # Tigre expects (Angle, DetectorV, DetectorU) -> (N, H, W)
        # Original script: projs[:, ::-1, :] implies flipping vertical axis?
        # Often CT data needs flip. Let's keep it consistent with original script which flips.
        
        # Verify angles match projections count
        if len(angles) != projs.shape[0]:
             print(f"Error: Angles count {len(angles)} != Projections count {projs.shape[0]}")
             return

        ct_gt = algs.fdk(projs[:, ::-1, :], geo, angles)
        
        # Transpose to (D, H, W) or (Z, Y, X)? 
        # Original: ct_gt.transpose((2, 1, 0))
        # Tigre output is usually (Z, Y, X)
        ct_gt = ct_gt.transpose((2, 1, 0))
        ct_gt[ct_gt < 0] = 0
        np.save(ct_gt_save_path, ct_gt)

    # Metadata
    meta_data = {
        "scanner": scanner_cfg,
        "ct": "vol_gt.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
    }
    
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Data saved in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to folder containing .mhd and config.txt")
    parser.add_argument("--output", type=str, help="Path to output.")
    parser.add_argument("--proj_subsample", default=4, type=int, help="subsample projections pixels")
    parser.add_argument("--proj_rescale", default=400.0, type=float, help="rescale projection values")
    parser.add_argument("--object_scale", default=50, type=int, help="Rescale scene")
    parser.add_argument("--n_test", default=100, type=int, help="number of test")
    parser.add_argument("--n_train", default=75, type=int, help="number of train")
    
    parser.add_argument("--nVoxel", nargs="+", default=[256, 256, 256], type=int, help="voxel dimension")
    parser.add_argument("--sVoxel", nargs="+", default=[2.0, 2.0, 2.0], type=float, help="volume size")
    parser.add_argument("--offOrigin", nargs="+", default=[0.0, 0.0, 0.0], type=float, help="offOrigin")
    parser.add_argument("--offDetector", nargs="+", default=[0.0, 0.0], type=float, help="offDetector")
    parser.add_argument("--accuracy", default=0.5, type=float, help="accuracy")
    
    args = parser.parse_args()
    main(args)