import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pred_path = "MRIdata/outputs/exp_0316_2032_iter1000_L2loss_under/point_cloud/iteration_1000/vol_pred.npy"
gt_path = "MRIdata/outputs/exp_0316_2032_iter1000_L2loss_under/point_cloud/iteration_1000/vol_gt.npy"
gt_unsampled_path = 'MRIdata/under/vol_gt_unsampled.npy'

recon_pocs = np.load('MRIdata/under/pocs_recon.npy')

gt = np.load('MRIdata/under/vol_gt.npy')
gt_unsampled = np.load('MRIdata/under/vol_gt_unsampled.npy')
pred = np.load('MRIdata/outputs/exp_0317_1558_iter1000_L2loss_under/point_cloud/iteration_1000/vol_pred.npy')

# setup show way ("slice" or "volume")
show_way = "volume"
use_pocs = True

if show_way == "volume":

    plotter = pv.Plotter(shape=(1,4), window_size=(3200,800))

    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth")
    plotter.add_volume(gt, cmap="viridis")  # 可以指定colormap
    
    # if use_pocs:
    # # 右子图显示pred
    plotter.subplot(0, 1)
    plotter.add_text("Recon_POCS")
    plotter.add_volume(recon_pocs, cmap="viridis")
    
# else:
    plotter.subplot(0, 2)
    plotter.add_text("Ground Truth(Unsampled)")
    plotter.add_volume(gt_unsampled, cmap="viridis")

    plotter.subplot(0, 3)
    plotter.add_text("Prediction")
    plotter.add_volume(pred, cmap="viridis")

    plotter.show()

elif show_way == "slice":
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(gt[130,:,:])
    plt.title("GT")

    plt.subplot(1,2,2)
    plt.imshow(pred[130,:,:])
    plt.title("PRED")

    plt.show()

else:
    print("please ensure show_way in ['slice', 'volume'].")