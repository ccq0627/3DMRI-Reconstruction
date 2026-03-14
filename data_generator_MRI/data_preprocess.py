import numpy as np
import nibabel as nib
import os.path as osp
from argparse import ArgumentParser
import sys
import json

def main(args):
    data_path = args.path
    dir_path = osp.dirname(data_path) if args.output is None else args.output
    save_path = osp.join(dir_path, "vol_gt.npy")
    
    nii_img = nib.ni1.load(data_path)
    data_5d = nii_img.get_fdata()
    data = data_5d[0, 0, ...]

    affine = nii_img.affine

    data = np.clip(data, 0, None)
    p_99_5 = np.percentile(data, 99.5)

    vol_gt = np.clip(data, 0, p_99_5) / p_99_5
    
    offOrigin = affine[:3, 3]
    nVoxel = np.array(vol_gt.shape)
    dVoxel = nii_img.header['pixdim'][3:6]
    sVoxel = nVoxel * dVoxel
    
    nii_data_path = osp.join(dir_path, "nii_data.json")
    nii_data = {
        "nii_cfg": {
            "offOrigin": offOrigin.tolist(),
            "nVoxel": nVoxel.tolist(),
            "dVoxel": dVoxel.tolist(),
            "sVoxel": sVoxel.tolist(),
        },
        "vol": "vol_gt.npy"
    }

    with open(nii_data_path,'w',encoding='utf-8') as f:
        json.dump(nii_data, f, indent=4, ensure_ascii=False)

    if not osp.exists(save_path):
        np.save(save_path, vol_gt)

    if False:
        import pyvista as pv
        vol_gt = np.load(save_path)
        plotter = pv.Plotter(window_size=(800,800), line_smoothing=True, off_screen=False)
        plotter.add_volume(vol_gt)
        plotter.show_axes()
        plotter.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to MRI data", default="MRIdata/00000.nii.gz")
    parser.add_argument("--output", type=str, help="Output folder", default=None)

    args = parser.parse_args()

    main(args)