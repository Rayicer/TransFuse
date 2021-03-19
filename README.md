# TransFuse
This repo holds the code of TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation

## Requirements
Pytorch>=1.6.0 (>=1.1.0 should work but not tested)
timm==0.3.2


## Model Overview
put transfuse figure here


## Experiments

### ISIC2017 Skin Lesion Segmentation Challenge
Any Nvidia GPU of memory>=4G shall be sufficient for this experiment. 

1. Preparing necessary data:
	+ downloading ISIC2017 training, validation and testing data from the [official site](https://challenge.isic-archive.com/data), put the unzipped data in `./data`.
	+ run 'process.py' to preprocess all the data, which generates `data_{train, val, test}.npy` and `mask_{train, val, test}.npy`.

2. Downloading pretrained models:
	+ downloading DeiT-small from [DeiT repo](https://github.com/facebookresearch/deit) to `./pretrained`.
	+ downloading resnet-34 from [timm Pytorch](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth) to `./pretrained`.
	+ (Optional) downloading our trained TransFuse-S from here to `./snapshots/`.

3. Testing:
	+ run `test_isic.py --ckpt_path='snapshots/TransFuse-8.pth'`

4. Training:
	+ run `train_isic.py`; you may also want to change the default saving path or other hparams.

## Reference
[Facebook DeiT](https://github.com/facebookresearch/deit)
[timm repo](https://github.com/rwightman/pytorch-image-models)
[PraNet repo](https://github.com/DengPingFan/PraNet)
[Hardnet-MSEG repo](https://github.com/james128333/HarDNet-MSEG)


## Citation
Please consider citing us if you find this work to be helpful:
@article{zhang2021transfuse,
  title={TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation},
  author={Zhang, Yundong and Liu, Huiye and Hu, Qiang},
  journal={arXiv preprint arXiv:2102.08005},
  year={2021}
}


## Questions
Please drop an email to huiyeliu@rayicer.com

