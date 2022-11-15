# iFormer: [Inception Transformer](http://arxiv.org/abs/2205.12956) (NeurIPS 2022 Oral)
This is a PyTorch implementation of iFormer proposed by our paper "[Inception Transformer](http://arxiv.org/abs/2205.12956)".


## Image Classification

### 1. Requirements
torch>=1.7.0; torchvision>=0.8.1; timm==0.5.4; fvcore; [apex-amp](https://github.com/NVIDIA/apex) (if you want to use fp16); 

data prepare: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Main results on ImageNet-1K

| Model      |  #params  | FLOPs | Image resolution | acc@1| Model |
| :---       |   :---:   |  :---: |  :---: |  :---:  |  :---:  |
| iFormer-S  |   20M     |   4.8G  |   224 |  83.4  | [model](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small.pth)/[config](https://github.com/sail-sg/iFormer/blob/main/checkpoint/iformer_small/args.yaml)/[log](https://github.com/sail-sg/iFormer/blob/main/checkpoint/iformer_small/summary.csv) |
| iFormer-B  |   48M     |   9.4G  |   224 |  84.6  | [model](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_base.pth)/[config](https://github.com/sail-sg/iFormer/blob/main/checkpoint/iformer_base/args.yaml)/[log](https://github.com/sail-sg/iFormer/blob/main/checkpoint/iformer_base/summary.csv) |
| iFormer-L  |   87M     |   14.0G |   224 |  84.8  | [model](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_large.pth)/[config](https://github.com/sail-sg/iFormer/blob/main/checkpoint/iformer_large/args.yaml)/[log](https://github.com/sail-sg/iFormer/blob/main/checkpoint/iformer_large/summary.csv) |

Fine-tuning Results with larger resolution (384x384) on ImageNet-1K

| Model      |  #params  | FLOPs | Image resolution | acc@1| Model |
| :---       |   :---:   |  :---: |  :---: |  :---:  |  :---:  |
| iFormer-S  |   20M     |   16.1G  |   384 |  84.6  | [model](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small_384.pth)/[config](https://github.com/sail-sg/iFormer/blob/main/checkpoint_384/iformer_small_384/args.yaml)/[log](https://github.com/sail-sg/iFormer/blob/main/checkpoint_384/iformer_small_384/summary.csv) |
| iFormer-B  |   48M     |   30.5G  |   384 |  85.7  | [model](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_base_384.pth)/[config](https://github.com/sail-sg/iFormer/blob/main/checkpoint_384/iformer_base_384/args.yaml)/[log](https://github.com/sail-sg/iFormer/blob/main/checkpoint_384/iformer_base_384/summary.csv) |
| iFormer-L  |   87M     |   45.3G  |   384 |  85.8  | [model](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_large_384.pth)/[config](https://github.com/sail-sg/iFormer/blob/main/checkpoint_384/iformer_large_384/args.yaml)/[log](https://github.com/sail-sg/iFormer/blob/main/checkpoint_384/iformer_large_384/summary.csv) |


### Training

Train iformer_small on 224
```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py /dataset/imagenet \
--model iformer_small -b 128 --epochs 300 --img-size 224 --drop-path 0.2 --lr 1e-3 \
--weight-decay 0.05 --aa rand-m9-mstd0.5-inc1 --warmup-lr 1e-6 --warmup-epochs 5 \
--output checkpoint --min-lr 1e-6 --experiment iformer_small
```

Finetune on 384 based on the pretrained checkpoint on 224
```bash
python -m torch.distributed.launch --nproc_per_node=8 fine-tune.py /dataset/imagenet \
--model iformer_small_384 -b 64 --lr 1e-5 --min-lr 1e-6 --warmup-lr 2e-8 --warmup-epochs 0 \
--epochs 20 --img-size 384 --drop-path 0.3 --weight-decay 1e-8 --mixup 0.1 --cutmix 0.1 \
--cooldown-epochs 10 --aa rand-m9-mstd0.5-inc1 --clip-grad 1.0 --output checkpoint_fine \
--initial-checkpoint checkpoint/iformer_small/model_best.pth.tar \
--experiment iformer_small_384
```

### Validation
```bash
python validate.py /dataset/imagenet --model iformer_small  --checkpoint checkpoint/iformer_small/model_best.pth.tar
```

## Object Detection and Instance Segmentation

All models are based on Mask R-CNN and trained by 1x  training schedule.

| Backbone  | #Param. | FLOPs | box mAP | mask mAP |
|:---------:|:-------:|:-----:|:-------:|:--------:|
| iFormer-S |   40M   |  263G |   46.2  |   41.9   |
| iFormer-B |    67M  |  351G |   48.3  |   43.3   |

## Semantic Segmentation

|  Backbone | Method  | #Param. | FLOPs | mIoU |
|:---------:|---------|:-------:|:-----:|:----:|
| iFormer-S | FPN     |   24M   |  181G | 48.6 |
| iFormer-S | Upernet |   49M   |  938G | 48.4 |

## Bibtex
```
@inproceedings{
si2022inception,
title={Inception Transformer},
author={Chenyang Si and Weihao Yu and Pan Zhou and Yichen Zhou and Xinchao Wang and Shuicheng YAN},
booktitle={Advances in Neural Information Processing Systems},
year={2022}
}
```

## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).


Besides, Weihao Yu would like to thank TPU Research Cloud (TRC) program for the support of partial computational resources.
