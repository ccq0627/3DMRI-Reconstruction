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

import torch
import sys
from datetime import datetime
import numpy as np
import random
from torch import Tensor

def t2a(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_mask(size, per: float, sigma: int):
    np.random.seed(0)
    Nx, Ny, Nz = size[0], size[1], size[2]
    total_lines = int(per * (Ny * Nz))

    y = np.arange(-Ny//2, Ny//2)
    z = np.arange(-Nz//2, Nz//2)
    ky, kz = np.meshgrid(y, z, indexing='ij')
    r = np.sqrt(ky**2 + kz**2)

    # define gaussian distribution
    probability_map = np.exp(-0.5 * (r**2) / (sigma**2))

    # set up full sample area
    acs_size = 32 # 设定中心 32x32 为全采样区
    acs_mask = (np.abs(ky) < acs_size//2) & (np.abs(kz) < acs_size//2)

    probability_map[acs_mask] = 0.0

    lines_needed = total_lines - np.sum(acs_mask)

    prob_flat = probability_map.flatten()
    prob_flat = prob_flat / np.sum(prob_flat) 

    # 不重复抽样
    chosen_indices = np.random.choice(
        np.arange(Ny * Nz), 
        size=lines_needed, 
        replace=False, 
        p=prob_flat
    )
    # generate mask
    mask_2d = acs_mask.flatten()
    mask_2d[chosen_indices] = True
    mask_2d = mask_2d.reshape((Ny, Nz)) # 还原回二维

    mask_3d = np.zeros((Nx, Ny, Nz), dtype=bool)
    mask_3d[:, mask_2d] = True

    return mask_3d

def fft(vol_3D: Tensor) -> Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(vol_3D), norm='ortho')
    )

def ifft(kspace_3D: Tensor) -> Tensor:
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(kspace_3D), norm='ortho')
    )

def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
