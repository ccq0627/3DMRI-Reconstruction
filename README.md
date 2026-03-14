### Introduction

This is a repo of MRI reconstruction with 3DGS.

## 1. Installation

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment. We tested the code on Ubuntu 20.04 with an RTX 3090 GPU. For installation issues on other platforms, please refer to [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```sh
# Download code
git clone https://github.com/ccq0627/3DMRI-Reconstruction.git --recursive

# Install environment
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate r2_gaussian

```

## 2. Dataset

```sh
└── MRIdata
    └── XXX.nii.g
```

## 3. Running

### 3.1 Initialization
you need to use `data_preprocess.py` to preprocess original data.Then, you need to use `initialize_pcd_MRI.py` to generate a `*.npy` file which stores the point cloud for Gaussian initialization.

### 3.2 Training

Use `train_MRI.py` to train Gaussians. Make sure that the initialization file `*.npy` has been generated.

```sh
# Training
python train.py
```

## 4. Acknowledgement, license and citation

Our code is adapted from [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF), [NAF](https://github.com/Ruyi-Zha/naf_cbct), [TIGRE toolbox](https://github.com/CERN/TIGRE.git) and [R2-Gausssian](https://github.com/Ruyi-Zha/r2_gaussian). We thank the authors for their excellent works.

This project is under the license of [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
