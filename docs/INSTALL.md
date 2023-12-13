## Installation

It is recommended to create three seperate virtual conda environments for pre-training, segmentation, and detection.

#### Pre-training

1. Install PyTorch 1.7.0+ and torchvision 0.8.1+ :

```bash
conda install -c pytorch pytorch torchvision
```

2. Install other packages:

```bash
pip install -r pretraining/requirements.txt
```

#### Segmentation

1. Install Pytorch 1.8.0 and torchvision 0.9.0:

```bash
conda install -c pytorch pytorch torchvision
```

2. Install the mmsegmentation library:

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
```

3. Install other packages:

```bash
pip install -r segmentation/requirements.txt
```

4. Install apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--global-option=--cpp_ext" --config-settings "--global-option=--cuda_ext" ./
```

#### Detection

Install detectron2 following the guide in [official detectron2 repository](https://github.com/facebookresearch/detectron2).