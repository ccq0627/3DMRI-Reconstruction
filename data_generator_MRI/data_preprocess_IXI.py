import numpy as np
import nibabel as nib
import os.path as osp
import os
from argparse import ArgumentParser
import sys
import json
import gc

sys.path.append("./")
from r2_gaussian.utils.general_utils import get_mask
from r2_gaussian.arguments import ModelParams

def fft(x):
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(x), norm='ortho'
        )
    )

def ifft(k):
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(k), norm='ortho'
        )
    )

def main(args, lp: ModelParams):
    """
    get kspace and niicfg
    """
    data_path = args.path
    accelerate_factor = lp.accelerate_factor
    dir_path = osp.join(osp.dirname(data_path),f"acc_rate{accelerate_factor}_sigma{lp.mask_sigma}")
    os.makedirs(dir_path, exist_ok=True)
    ks_save_path = osp.join(dir_path, "kspace.npy")  # undersampled kspace 
    vol_ifft_save_path = osp.join(dir_path, "vol_ifft.npy")  # IFFT
    vol_save_path = osp.join(dir_path, "vol_gt.npy")
    mask_save_path = osp.join(dir_path, "sample_mask.npy")
    
    nii_img = nib.ni1.load(data_path)
    data = np.array(nii_img.dataobj[:,:,:], dtype=np.float32).transpose(1,0,2)
    affine = nii_img.affine

    np.clip(data, 0, None, out=data)
    p_99_5 = np.percentile(data, 99.5)
    np.clip(data, 0, p_99_5, out=data)
    data /= p_99_5
    vol_gt = data
    
    offOrigin = affine[:3, 3]
    nVoxel = np.array(vol_gt.shape)
    dVoxel = nii_img.header['pixdim'][1:4]
    sVoxel = nVoxel * dVoxel
    
    nii_data_path = osp.join(dir_path, "nii_data.json")
    nii_data = {
        "nii_cfg": {
            "offOrigin": offOrigin.tolist(),
            "nVoxel": nVoxel.tolist(),
            "dVoxel": dVoxel.tolist(),
            "sVoxel": sVoxel.tolist(),
        },
        "vol": "vol_gt.npy",
        "vol_ifft": "vol_ifft.npy",
        "kspace": "kspace.npy",
        "mask": "sample_mask.npy",
    }
    with open(nii_data_path,'w',encoding='utf-8') as f:
        json.dump(nii_data, f, indent=4, ensure_ascii=False)

    np.save(vol_save_path, vol_gt)
    # get kspace data full
    kspace_full = fft(vol_gt)
    kspace_full = kspace_full.astype(np.complex64)

    del vol_gt, data
    gc.collect()

    # get mask
    mask_3d = get_mask(size=nVoxel, per=1.0/accelerate_factor, sigma=lp.mask_sigma)
    np.save(mask_save_path, mask_3d)
    
    # get kspace undersampled
    kspace_undersampled = kspace_full * mask_3d  # complex
    
    del kspace_full
    gc.collect()

    np.save(ks_save_path, kspace_undersampled)
    # IFFT 
    vol_gt_undersampled = np.abs(ifft(kspace_undersampled))

    np.save(vol_ifft_save_path, vol_gt_undersampled)
    del vol_gt_undersampled
    gc.collect()

    print(f"Data preprocessing completed. Files saved in {dir_path}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--path", type=str, help="Path to MRI data", default="MRIdata/IXI/IXI002-Guys-0828-T1.nii.gz")

    args = parser.parse_args()

    main(args, lp)