import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

root_dir = "data/real_dataset/cone_ntrain_50_angle_360" 
path = osp.join(root_dir, "seashell/proj_train/0001.npy")
data = np.load(path)
max = np.max(data)
min = np.min(data)
data1 = (data - min) / (max - min)
plt.figure(figsize=(6,6))
plt.imshow(data,cmap="gray")
plt.axis('off')
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(data1,cmap="gray")
plt.axis('off')
plt.show()

print(data.shape)
print(osp.basename(path))
