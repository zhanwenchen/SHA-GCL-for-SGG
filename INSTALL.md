## Installation

Most of the requirements of this projects are exactly the same as [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, we recommend you to check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python >= 3.6 (Mine 3.6)
- PyTorch >= 1.2 (Mine 1.6.0 (CUDA 10.1))
- torchvision >= 0.4 (Mine 0.7.0 (CUDA 10.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

# this installs the right pip and dependencies for the fresh python
conda create --name gcl python=3.6 ipython scipy h5py
conda activate gcl

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python-headless overrides

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
# pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# The closest pytorch version that supports 11.0 is 1.7.0
# conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
# OSError: /home/zhanwen/miniconda3/envs/SHA_GCL/lib/python3.6/site-packages/torch/lib/../../../../libcublas.so.11: undefined symbol: free_gemm_select, version libcublasLt.so.11
# the lowest version using cuda 11.1 is 1.8.0
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# 1.8.0 has a bug: File "/home/zhanwen/gcl/maskrcnn_benchmark/modeling/balanced_positive_negative_sampler.py", line 50, in __call__
#     perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
# RuntimeError: radix_sort: failed on 1st step: cudaErrorInvalidDevice: invalid device ordinal

# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# 1.8.1 has cuda availability problems:
# python
# import torch
# torch.version.cuda None
# torch.cuda.is_available() False
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia


export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
pip install packaging
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 22.04-dev
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
# cd $INSTALL_DIR
# git clone https://github.com/zhanwenchen/SHA-GCL-for-SGG.git
# cd SHA-GCL-for-SGG

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
# ValueError: Unknown CUDA arch (8.6) or GPU not supported

unset INSTALL_DIR

# change dataset path in dataset_path.py
# datasets_path = '/home/zhanwen/'

# create a symlink vg that points to visual_genome folder
# Copy glove.6B files to GLOVE_DIR ./datasets/vg/
