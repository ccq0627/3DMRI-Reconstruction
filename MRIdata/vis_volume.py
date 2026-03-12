import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pred_path = "MRIdata/output_2026_3_6/point_cloud/iteration_1000/vol_pred.npy"
gt_path = "MRIdata/output_2026_3_6/point_cloud/iteration_1000/vol_gt.npy"
gt = np.load(gt_path)
pred = np.load(pred_path)

# setup show way ("slice" or "volume")
show_way = "volume"

if show_way == "volume":

    plotter = pv.Plotter(shape=(1,2), window_size=(1600,800))

    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth")
    plotter.add_volume(gt, cmap="viridis")  # 可以指定colormap

    # 右子图显示pred
    plotter.subplot(0, 1)
    plotter.add_text("Prediction")
    plotter.add_volume(pred, cmap="viridis")

    plotter.show()

elif show_way == "slice":
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(gt[260,:,:])
    plt.title("GT")

    plt.subplot(1,2,2)
    plt.imshow(pred[260,:,:])
    plt.title("PRED")

    plt.show()

else:
    print("please ensure show_way in ['slice', 'volume'].")