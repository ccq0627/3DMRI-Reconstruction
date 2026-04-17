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

def fft(image):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image), norm='ortho'))

def ifft(kspace):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace), norm='ortho'))

def main(args, lp: ModelParams):
    """
    get kspace and niicfg
    """
    data_path = args.path
    if lp.accelerate_factor is not None:
        accelerate_factor = lp.accelerate_factor
    dir_path = osp.join(osp.dirname(data_path),f"acc_rate{accelerate_factor}_sigma{lp.mask_sigma}")
    os.makedirs(dir_path, exist_ok=True)
    ks_save_path = osp.join(dir_path, "kspace_gt.npy")  # 欠采样kspace 
    vol_unsampled_save_path = osp.join(dir_path, "vol_gt_unsampled.npy")  # IFFT
    vol_save_path = osp.join(dir_path, "vol_gt.npy")
    mask_save_path = osp.join(dir_path, "mask_3D.npy")
    
    nii_img = nib.ni1.load(data_path)
    data = np.array(nii_img.dataobj[:,:,:], dtype=np.float32).transpose(1,0,2)
    # data = np.array(nii_img.dataobj[0,0,:,:,:], dtype=np.float32)
    affine = nii_img.affine

    np.clip(data, 0, None, out=data)
    p_99_5 = np.percentile(data, 99.5)
    np.clip(data, 0, p_99_5, out=data)
    data /= p_99_5
    vol_gt = data
    
    offOrigin = affine[:3, 3]
    nVoxel = np.array(vol_gt.shape)
    dVoxel = nii_img.header['pixdim'][1:4]
    # dVoxel = nii_img.header['pixdim'][3:6]
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
        "vol_unsampled": "vol_gt_unsampled.npy",
        "vol_kspace": "kspace_gt.npy",
        "mask_3D": "mask_3D.npy",
        "accelerate_factor": accelerate_factor,
    }
    with open(nii_data_path,'w',encoding='utf-8') as f:
        json.dump(nii_data, f, indent=4, ensure_ascii=False)

    np.save(vol_save_path, vol_gt)
    # get kspace data full
    kspace_full = fft(vol_gt)
    kspace_full = kspace_full.astype(np.complex64) # 强制转为单精度复数，内存减半！

    # kspace_full = np.fft.fftshift(
    #     np.fft.fftn(np.fft.ifftshift(vol_gt), norm='ortho')
    # )
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
    # IFFT 用于可视化和采样点
    vol_gt_undersampled = ifft(kspace_undersampled)
    vol_gt_undersampled_mag = np.abs(vol_gt_undersampled).astype(np.float32)

    del vol_gt_undersampled # 释放复数矩阵
    gc.collect()

    vol_gt_undersampled_mag /= np.max(vol_gt_undersampled_mag)
    np.save(vol_unsampled_save_path, vol_gt_undersampled_mag)
    del vol_gt_undersampled_mag
    gc.collect()

    print(f"Data preprocessing completed. Files saved in {dir_path}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--path", type=str, help="Path to MRI data", default="MRIdata/IXI002-Guys-0828-T1.nii.gz")
    # parser.add_argument("--path", type=str, help="Path to MRI data", default="MRIdata/00000.nii.gz")
    # parser.add_argument("--accelerate_factor", type=int, help="Accelerate factor", default=2)
    
    args = parser.parse_args()

    main(args, lp)