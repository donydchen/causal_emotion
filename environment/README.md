# IERN Environment Installation


## Manual installation

The provided exported anoconda environment file is mainly tested on `NVIDIA GeForce RTX 2080 Ti` with cuda 10. If it does not work on your machine, please consider installing the environment manually by following the instructions below. 

* initialize a python 3.8 environment

```bash
conda create -n iern python=3.8.2
conda activate iern
```

* install pytorch 1.4.0. Choose a correct CUDA version from [here](https://pytorch.org/get-started/previous-versions/#v140). Below we provide the commands for cuda10

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

* install related python dependencies

```bash
pip install scikit-learn==0.23.2 matplotlib==3.1.3 tqdm==4.45.0 seaborn==0.11.0 tensorboard==2.7.0
pip install protobuf==3.19.0  # fix bug for tensorboard
pip install imutils==0.5.3 opencv-python==4.3.0.36 dlib==19.20.0  # for preprocessing the input dataset
```

## Miscs

Please follow the instructions to use PyTorch 1.4.0, it might encounter some unexpected bugs if the PyTorch version is higher than the recommended version.

