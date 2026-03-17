import numpy as np
import nibabel as nib
import os.path as osp
import os
from argparse import ArgumentParser
import sys
import json
from scipy.ndimage import gaussian_filter
import gc
import scipy.fft as sfft

sys.path.append("./")
from r2_gaussian.utils.general_utils import get_mask
from POCS import pocs_reconstruction


def main(args):
    """
    get kspace and niicfg
    """
    data_path = args.path
    dir_path = osp.join(osp.dirname(data_path),"full") if args.model == "full" else osp.join(osp.dirname(data_path),"under")
    os.makedirs(dir_path, exist_ok=True)
    ks_save_path = osp.join(dir_path, "kspace_gt.npy")
    vol_unsampled_save_path = osp.join(dir_path, "vol_gt_unsampled.npy")
    vol_save_path = osp.join(dir_path, "vol_gt.npy")
    mask_save_path = osp.join(dir_path, "mask_3D.npy")
    
    nii_img = nib.ni1.load(data_path)
    data = np.array(nii_img.dataobj[0, 0, ...], dtype=np.float32)

    affine = nii_img.affine

    np.clip(data, 0, None, out=data)
    p_99_5 = np.percentile(data, 99.5)
    np.clip(data, 0, p_99_5, out=data)
    data /= p_99_5
    vol_gt = data
    
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
        "vol": "vol_gt.npy",
        "vol_unsampled": "vol_gt_unsampled.npy",
        "vol_kspace": "kspace_gt.npy",
        "mask_3D": "mask_3D.npy",
        "mode": str(args.model)
    }
    with open(nii_data_path,'w',encoding='utf-8') as f:
        json.dump(nii_data, f, indent=4, ensure_ascii=False)

    np.save(vol_save_path, vol_gt)
    # get kspace data full
    kspace_full = sfft.fftshift(sfft.fftn(sfft.ifftshift(vol_gt), norm='ortho', workers=-1))
    kspace_full = kspace_full.astype(np.complex64) # 强制转为单精度复数，内存减半！

    # kspace_full = np.fft.fftshift(
    #     np.fft.fftn(np.fft.ifftshift(vol_gt), norm='ortho')
    # )
    del vol_gt, data
    gc.collect()

    # get mask
    if args.model == "full":
        mask_3d = get_mask(size=nVoxel,per=1.0)
    elif args.model == "under":
        mask_3d = get_mask(size=nVoxel,per=0.1)
    np.save(mask_save_path, mask_3d)
    # get kspace undersampled
    kspace_undersampled = kspace_full * mask_3d  # complex
    
    del kspace_full
    gc.collect()

    np.save(ks_save_path, kspace_undersampled)
    # 伪gt 用于可视化 和采样点
    # vol_gt_undersampled = np.fft.fftshift(
    #     np.fft.ifftn(np.fft.ifftshift(kspace_undersampled), norm='ortho')
    # )
    # vol_gt_undersampled_mag = np.abs(vol_gt_undersampled)  # float
    # vol_gt_undersampled_mag = vol_gt_undersampled_mag / np.max(vol_gt_undersampled_mag)
    vol_gt_undersampled = sfft.fftshift(sfft.ifftn(sfft.ifftshift(kspace_undersampled), norm='ortho', workers=-1))
    vol_gt_undersampled_mag = np.abs(vol_gt_undersampled).astype(np.float32)

    del vol_gt_undersampled # 释放复数矩阵
    gc.collect()

    vol_gt_undersampled_mag /= np.max(vol_gt_undersampled_mag)
    np.save(vol_unsampled_save_path, vol_gt_undersampled_mag)
    del vol_gt_undersampled_mag
    gc.collect()

    recon_pocs = pocs_reconstruction(kspace_undersampled,mask_3d)
    
    recon_denoised = gaussian_filter(recon_pocs, sigma=1.5)  # 高斯去噪

    del recon_pocs
    gc.collect()

    recon_final = recon_denoised / np.max(recon_denoised)  # POCS重建结果，用于初始化点云
    pocs_recon_save_path = osp.join(dir_path, "pocs_recon.npy")

    np.save(pocs_recon_save_path, recon_final)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to MRI data", default="MRIdata/00000.nii.gz")
    parser.add_argument("--model", type=str, help="Sample model(full or under)", default="under")

    args = parser.parse_args()

    main(args)