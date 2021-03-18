# TransFuse
This repo holds the code of TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation

## Requirements
Pytorch>=1.6.0 (>=1.1.0 should work but not tested)
timm==0.3.2


## Model Overview
put transfuse figure here


## Experiments

### ISIC2017 Skin Lesion Segmentation Challenge
Any Nvidia GPU of memory>=4G shall be sufficient for this experiment

1. Preparing necessary data:
	+ downloading ISIC2017 training, validation and testing data from the [download link (official site)](https://challenge.isic-archive.com/data), put the unzipped data in `./data`;
	+ run 'process.py' to preprocess all the data, which generates `data_{train, val, test}.npy` and `mask_{train, val, test}.npy`.

2. Downloading pretrained models:
	+ downloading DeiT-Small from facebook repo;
	+ downloading resnet-34 from timm Pytorch repo;




