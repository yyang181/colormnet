# ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization

### [Project Page](https://yyang181.github.io/colormnet/) | [Paper (ArXiv)](https://arxiv.org/abs/2404.06251) | [Supplemental Material](https://arxiv.org/abs/2404.06251) | [Code (Github)](https://github.com/yyang181/colormnet) 

[![google colab logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1naXws0elPMunfcvKSryLW1lFnPOF6Nb-?usp=sharing) [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/yyang181/ColorMNet) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=yyang181/colormnet)[![GitHub Stars](https://img.shields.io/github/stars/yyang181/colormnet?style=social)](https://github.com/yyang181/colormnet)


**This repository is the official pytorch implementation of our paper, *ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization*.**

[Yixin Yang](https://imag-njust.net/),
[Jiangxin Dong](https://imag-njust.net/),
[Jinhui Tang](https://imag-njust.net/jinhui-tang/),
[Jinshan Pan](https://jspan.github.io/) <br>

Nanjing University of Science and Technology

## 🔥 News
<!-- - [2024-09-01] Integrated to :hugs: [Hugging Face](https://huggingface.co/spaces). Try out online demo! [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/yyang181/ColorMNet) -->
- [2024-09-09] Add training code, see [train.py](https://github.com/yyang181/colormnet/blob/main/train.py).
- [2024-09-09] Colab demo for ColorMNet is available at [![google colab logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1naXws0elPMunfcvKSryLW1lFnPOF6Nb-?usp=sharing).
- [2024-09-07] Add inference code and pretrained weights, see [test.py](https://github.com/yyang181/colormnet/blob/main/test.py).
- [2024-04-13] Project page released at [ColorMNet Project](https://yyang181.github.io/ColorMNet). Please be patient and stay updated.

## Requirements

* Python 3.8+
* PyTorch 1.11+ (See [PyTorch](https://pytorch.org/) for installation instructions)
* `torchvision` corresponding to the PyTorch version
* OpenCV (try `pip install opencv-python`)
* Others: `pip install -r requirements.txt`

## :briefcase: Dependencies and Installation

```
# git clone this repository

conda create -n colormnet python=3.8 
conda activate colormnet 

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# install py-thin-plate-spline
git clone https://github.com/cheind/py-thin-plate-spline.git
cd py-thin-plate-spline && pip install -e . && cd ..

# install Pytorch-Correlation-extension
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git 
cd Pytorch-Correlation-extension && python setup.py install && cd ..

pip install -r requirements.txt
```

#### 

## :gift: Checkpoints

Download the pretrained models manually and put them in `./saves` (create the folder if it doesn't exist).

|   Name    |                             URL                              |
| :-------: | :----------------------------------------------------------: |
| ColorMNet | [model](https://drive.google.com/file/d/1PNxrNSoO0Uf8JeDkn3IZ9ouW1Y72jWaD/view?usp=share_link) |

## :zap: Quick Inference

- **Test on Images**: 

```
CUDA_VISIBLE_DEVICES=0 python test.py
```

## Train
### Dataset structure for both the training set and the validation set
```
# Specify --davis_root and --validation_root
data_root/
├── 001/
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── 002/
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── ...
```
### Training script
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --master_port 25205 \
    --nproc_per_node=1 \
    train.py \
    --exp_id DINOv2FeatureV6_LocalAtten_NTIRE2023dataset \
    --davis_root /path/to/your/training/data/\
    --validation_root /path/to/your/validation/data\
    --savepath ./wandb_save_dir
```

### To Do
- [x] Release training code
- [x] Release testing code
- [x] Release pre-trained models
- [x] Release demo

### Citation
If our work is useful for your research, please consider citing:

    @inproceedings{yang2024colormnet,
        author = {Yang, Yixin and Dong, Jiangxin and Tang, Jinhui and Pan Jinshan},
        title = {ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization},
        booktitle = {ECCV},
        year = {2024}
    }

### License

This project is licensed under <a rel="license" href="https://github.com/yyang181/colormnet/blob/main/LICENSE">BY-NC-SA 4.0</a>, while some methods adopted in this project are with other licenses. Please refer to [LICENSES.md](https://github.com/yyang181/colormnet/blob/main/LICENSES.md) for the careful check. Redistribution and use should follow this license.

### Acknowledgement

This project is based on [XMem](https://github.com/hkchengrex/XMem). Some codes are brought from [DINOv2](https://github.com/facebookresearch/dinov2). Thanks for their awesome works.

### Contact

This repo is currently maintained by Yixin Yang ([@yyang181](https://github.com/yyang181)) and is for academic research use only. 

