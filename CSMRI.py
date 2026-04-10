import numpy as np
import pywt
import pyvista as pv
import os.path as osp
import os

from time import time


# image -> kspace
def fft(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(x), norm="ortho")
    )

# kspace -> image
def ifft(y: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(y), norm="ortho")
    )

def data_consistency(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    k = fft(x)  # kspace
    k = mask * y + (1 - mask) * k
    return ifft(k)  # image

def wavelet_transform(x: np.ndarray, lam: float, wavelet='db4', level=2):
    x_real = np.real(x)

    coeffs = pywt.wavedecn(x_real, wavelet=wavelet, level=level)

    def soft_thresholding(coeffs, lam):
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - lam, 0)
    
    coeffs_thresh = []
    for i, c in enumerate(coeffs):
        if i == 0:
            coeffs_thresh.append(c)  # Approximation coefficients are not thresholded
        else:
            c_thresh = {k: soft_thresholding(v, lam) for k, v in c.items()}
            coeffs_thresh.append(c_thresh)

    x_recon = pywt.waverecn(coeffs_thresh, wavelet=wavelet)

    return x_recon

def cs_mri_3d_recon(
        y: np.ndarray, 
        mask: np.ndarray, 
        max_iter: int = 3000,
        tol: float = 1e-6,
        lam: float = 0.01,
        verbose: bool = True,
) -> np.ndarray:
    x = ifft(y)  # Initial zero-filled reconstruction

    prev_x = x.copy()
    for i in range(max_iter):

        x = data_consistency(x, y, mask)  # Data consistency step
        x = wavelet_transform(x, lam)  # Wavelet sparsity step

        # Check for convergence
        diff = np.linalg.norm(x - prev_x) / (np.linalg.norm(prev_x) + 1e-8)
        if verbose:
            if (i + 1) % 50 == 0 :
                print(f"Iteration {i+1}, Relative Change: {diff:.6f}")
        
        if diff < tol:
            break
        prev_x = x.copy()

    return np.abs(x)  # Return the magnitude of the reconstructed image


kspace = np.load("MRIdata/acc_rate2/kspace_gt.npy")
mask = np.load("MRIdata/acc_rate2/mask_3D.npy")

start_time = time()

recon = cs_mri_3d_recon(kspace, mask)

end_time = time()
print(f"Reconstruction completed in {end_time - start_time:.2f} seconds.")

if not osp.exists("CSMRI_results"):
    os.makedirs("CSMRI_results")

np.save("CSMRI_results/recon_2acc.npy", recon)
print(f"CS-MRI Reconstruction saved to CSMRI_results/recon_2acc.npy")

plotter = pv.Plotter(window_size=(800, 800),line_smoothing=True)

plotter.add_volume(recon, cmap="viridis")

plotter.add_axes()

plotter.show()