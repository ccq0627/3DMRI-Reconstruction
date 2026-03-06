import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss

a = torch.arange(8).reshape(2,2,2)
b = torch.full(size=(2,2,2),fill_value=2)
print(a,b)
print(ssim(a,b))

