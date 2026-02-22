# This is a demo script to plot 3D volume images in Fig. 1 of our paper. A monitor is required.

import numpy as np
import pyvista as pv

# Path to *.npy volume file
volume_path = "data/real_dataset/cone_ntrain_50_angle_360/seashell/output/point_cloud/iteration_30000/vol_pred.npy"
# volume_path = "data/real_dataset/cone_ntrain_50_angle_360/seashell/vol_gt.npy"
# Image save path
save_path = "volume_pred_z.png"

# Pyvista settings. You may need to tune them.
cpos = [
    (-458.0015547298666, -207.26124611865254, 324.4699978427509),
    (129.02644270914504, 111.50694084289574, 98.55158287937994),
    (0.0, 0.0, 79.59633400474613),
]
window_size = [800, 1000]
colormap = "viridis"

volume = np.load(volume_path)
half_size = volume.shape[2] // 2
# volume[:, :, half_size+1:] = 0  # Set half to zero to show the inner structure
# volume[:, :, :half_size] = 0
# volume[:, half_size+1:, :] = 0  # Set half to zero to show the inner structure
# volume[:, :half_size, :] = 0
# volume[half_size+1:, :, :] = 0  # Set half to zero to show the inner structure
volume[:half_size, :, :] = 0
clim = [0.0, 1.0]

plotter = pv.Plotter(window_size=window_size, line_smoothing=True, off_screen=True)
plotter.add_volume(volume, cmap="gray", opacity="linear", clim=clim)
plotter.camera_position = cpos
plotter.show(screenshot=save_path)
