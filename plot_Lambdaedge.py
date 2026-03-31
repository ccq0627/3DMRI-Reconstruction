import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def create_comparison_figure(image_paths, labels, roi_list, save_path="comparison_result_baseline_150.png"):
    """
    image_paths: 图片路径列表
    labels: 图片右下角的数值/标签列表
    roi_list: ROI 区域坐标列表 [y, x, h, w] (可以定义多个 ROI)
    """
    num_images = len(image_paths)
    num_rois = len(roi_list)

    # 根据首张图的宽高比自适应画布，避免非方图（如 150x256）显示过紧
    sample_img = Image.open(image_paths[0]).convert('L')
    sample_arr = np.array(sample_img)
    img_h, img_w = sample_arr.shape
    aspect = img_w / img_h
    
    # 设置色彩映射 (医学影像常用 'viridis' 或 'gray')
    cmap = 'viridis' 
    
    # 创建画布：第一行放主图，后面几行放 ROI 放大图
    # gridspec_kw 用于调整行高比例，主图大，ROI 小
    main_row_h = 4.0
    roi_row_h = 1.12
    per_image_w = np.clip(main_row_h * aspect, 2.4, 4.2)
    fig_w = num_images * per_image_w
    fig_h = main_row_h + roi_row_h

    fig, axes = plt.subplots(
        2,
        num_images,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': [main_row_h, roi_row_h]}
    )
    plt.subplots_adjust(wspace=0.03, hspace=0.03) # 非方图时适度留白

    roi_colors = ['red', 'dodgerblue', 'green'] # 对应不同 ROI 的颜色

    for i in range(num_images):
        # 1. 加载图片 (这里假设是单通道灰度图)
        img = Image.open(image_paths[i]).convert('L')
        img_arr = np.array(img)

        # --- 上方：显示主图 ---
        ax_main = axes[0, i]
        ax_main.imshow(img_arr, cmap=cmap)
        ax_main.axis('off')

        # 在第一张图左上角写 "Chest" (仿照示例图)
        if i == 0:
            ax_main.text(5, 30, 'Brain', color='white', fontsize=14, fontweight='bold')

        # 右下角写数值（统一锚点，多行文本可保证对齐）
        metric_lines = [labels['SSIM'][i], labels['PSNR'][i]]
        metric_text = "\n".join([m for m in metric_lines if m])
        ax_main.text(
            img_arr.shape[1] - 5,
            img_arr.shape[0] - 210,
            metric_text,
            color='white',
            fontsize=12,
            ha='right',
            va='bottom',
            linespacing=1.25,
            fontfamily='DejaVu Sans Mono'
        )

        # 在主图上绘制 ROI 矩形框
        for r_idx, (ry, rx, rh, rw) in enumerate(roi_list):
            rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2, 
                                     edgecolor=roi_colors[r_idx], facecolor='none')
            ax_main.add_patch(rect)

        # --- 下方：显示 ROI 放大图 ---
        # 我们把多个 ROI 拼在一个 subplot 里，或者调整布局
        # 为了模仿示例图，我们创建两个并排的小图
        ax_roi = axes[1, i]
        ax_roi.axis('off')
            
        # 进阶做法：在 axes[1, i] 的位置手动创建子区域展示所有 ROI
        for r_idx, (ry, rx, rh, rw) in enumerate(roi_list):
            roi_img = img_arr[ry:ry+rh, rx:rx+rw]
            # 计算子图位置：在 ax_roi 的范围内平分
            sub_ax = ax_roi.inset_axes([r_idx/num_rois , 0, 1/num_rois , 1])
            sub_ax.imshow(roi_img, cmap=cmap)
            # 设置 ROI 边框颜色
            for spine in sub_ax.spines.values():
                spine.set_edgecolor(roi_colors[r_idx])
                spine.set_linewidth(2)
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

    # 保存结果
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
    print(f"Figure saved to {save_path}")
    plt.show()


gt = np.load("MRIdata/outputs/exp_0326_2105_iter3000_6_wi/point_cloud/iteration_3000/vol_pred.npy")
pred_ssim0 = np.load('MRIdata/outputs/exp_0326_1501_iter3000_8_wi/point_cloud/iteration_3000/vol_pred.npy')
pred_edge001 = np.load('MRIdata/outputs/exp_0326_1519_iter3000_8_wi/point_cloud/iteration_3000/vol_pred.npy')
pred_edge005 = np.load('MRIdata/outputs/exp_0326_1534_iter3000_8_wi/point_cloud/iteration_3000/vol_pred.npy')
pred_edge01 = np.load('MRIdata/outputs/exp_0326_1554_iter3000_8_wi/point_cloud/iteration_3000/vol_pred.npy')
pred_edge02 = np.load('MRIdata/outputs/exp_0326_1607_iter3000_8_wi/point_cloud/iteration_3000/vol_pred.npy')
pred_edge05 = np.load('MRIdata/outputs/exp_0327_1755_iter3000_8_wi/point_cloud/iteration_3000/vol_pred.npy')

pred_edge05_slice = pred_edge05[150, :, :]
pred_edge02_slice = pred_edge02[150, :, :]
pred_edge01_slice = pred_edge01[150, :, :]
pred_edge005_slice = pred_edge005[150, :, :]
pred_edge001_slice = pred_edge001[150, :, :]
pred_edge0_slice = pred_ssim0[150, :, :]
gt_slice = gt[150, :, :]
# 保存切片为图片

Image.fromarray((np.clip(pred_edge05_slice, 0, 1) * 255).astype(np.uint8)).save("pred_edge05_slice_150_05.png")
Image.fromarray((np.clip(pred_edge02_slice, 0, 1) * 255).astype(np.uint8)).save("pred_edge02_slice_150_02.png")
Image.fromarray((np.clip(pred_edge01_slice, 0, 1) * 255).astype(np.uint8)).save("pred_edge01_slice_150_01.png")
Image.fromarray((np.clip(pred_edge005_slice, 0, 1) * 255).astype(np.uint8)).save("pred_edge005_slice_150_005.png")
Image.fromarray((np.clip(pred_edge001_slice, 0, 1) * 255).astype(np.uint8)).save("pred_edge001_slice_150_001.png")
Image.fromarray((np.clip(pred_edge0_slice, 0, 1) * 255).astype(np.uint8)).save("pred_edge0_slice_150_0.png")
Image.fromarray((gt_slice * 255).astype(np.uint8)).save("gt_slice_150.png")
# --- 使用示例 ---
# 假设你有 8 张图片
img_files = ["pred_edge05_slice_150_05.png", "pred_edge02_slice_150_02.png", "pred_edge01_slice_150_01.png", "pred_edge005_slice_150_005.png","pred_edge001_slice_150_001.png","pred_edge0_slice_150_0.png", "gt_slice_150.png", ]  # 这里替换成你的图片路径列表
# 对应的 PSNR 标签

metrics = {
    "PSNR": ["", "", "", "", "", "", ""],
    "SSIM": ["λedge=0.5", "λedge=0.2", "λedge=0.1", "λedge=0.05", "λedge=0.01", "λedge=0.0","Ground Truth"]
}  # psnr
# 定义两个 ROI 区域: [y, x, height, width]

rois = [
    [85, 50, 50, 50], # 红色框位置
    [190, 50, 50, 50]  # 蓝色框位置
]

# 注意：运行前请确保 img_files 里的路径真实存在
create_comparison_figure(img_files, metrics, rois, save_path=f"comparison_result_131_λ.png")