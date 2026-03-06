import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pred_path = "MRIdata/output_2026_3_6/point_cloud/iteration_1000/vol_pred.npy"
gt_path = "MRIdata/output_2026_3_6/point_cloud/iteration_1000/vol_gt.npy"
gt = np.load(gt_path)
pred = np.load(pred_path)
pred = pred / np.max(pred)

plotter = pv.Plotter(shape=(1,2), window_size=(1600,800))

plotter.subplot(0, 0)
plotter.add_text("Ground Truth")
plotter.add_volume(gt, cmap="viridis")  # 可以指定colormap

# 右子图显示pred
plotter.subplot(0, 1)
plotter.add_text("Prediction")
plotter.add_volume(pred, cmap="viridis")

plotter.show()