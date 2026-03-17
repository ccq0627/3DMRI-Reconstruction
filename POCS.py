import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from time import time
# kspace_undersampled_path = 'MRIdata/under/kspace_gt.npy'
# mask_path = 'MRIdata/under/mask_3D.npy'

# kspace_undersampled = np.load(kspace_undersampled_path)
# mask_3D = np.load(mask_path)

# recon = np.fft.fftshift(
#     np.fft.ifftn(np.fft.ifftshift(kspace_undersampled),norm='ortho')
# )
# recon_abs = np.abs(recon) / (np.abs(recon)).max()


def pocs_reconstruction(k_undersampled, mask, n_iterations=30, lambda_reg=0.005):
    """
    参数:
        k_undersampled: 欠采样的 K 空间数据 (复数, shape: 261x350x350)
        mask: 采样掩模 (布尔型, shape: 261x350x350)
        n_iterations: 迭代次数
        lambda_reg: 软阈值正则化参数
    """
    #  初始估计
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k_undersampled), norm='ortho'))
    
    for i in range(n_iterations):

        magnitude = np.abs(x)
        phase = np.exp(1j * np.angle(x)) # 提取并保留相位
        
        # 只对幅度减去 lambda_reg，保持非负
        magnitude_thresh = np.maximum(magnitude - lambda_reg, 0)
        
        # 将阈值化后的幅度与原相位重新结合
        x = magnitude_thresh * phase

        #  变换回 K 空间
        k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))

        #  数据一致性投影 (Data Consistency, DC)
        #  ~mask 表示未采样的位置，保留我们当前估计的高频推测值
        k = k * (~mask) + k_undersampled

        #  变换回图像域，准备下一次迭代
        x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k), norm='ortho'))

    return np.abs(x)

# start = time()
# recon_pocs = pocs_reconstruction(kspace_undersampled, mask_3D, n_iterations=30, lambda_reg=0.005)
# end = time()
# print(f"Spending time:{end-start} s")
# #  施加高斯滤波，抹平残余的高频伪影和噪声
# # sigma 的值 (如 0.8 到 1.5) 根据你的伪影严重程度调整
# recon_denoised = gaussian_filter(recon_pocs, sigma=1.5)

# recon_final_init = recon_denoised / np.max(recon_denoised)

# plotter = pv.Plotter(shape=(1,2),window_size=(1600,800))

# plotter.subplot(0, 0)
# plotter.add_text("recon")
# plotter.add_volume(recon_abs, cmap="viridis")  # 可以指定colormap

# # 右子图显示pred
# plotter.subplot(0, 1)
# plotter.add_text("recon_pocs")
# plotter.add_volume(recon_final_init, cmap="viridis")

# plotter.show()