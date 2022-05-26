# iFormer: [Inception Transformer](http://arxiv.org/abs/2205.12956)
This is a PyTorch implementation of iFormer proposed by our paper "[Inception Transformer](http://arxiv.org/abs/2205.12956)".

The code and model will be released soon.

## Image Classification

### Main results on ImageNet-1K

| Model      |  #params  | FLOPs | Image resolution | acc@1| Model |
| :---       |   :---:   |  :---: |  :---: |  :---:  |  :---:  |
| iFormer-S  |   20M     |   4.8G  |   224 |  83.4  | soon |
| iFormer-B  |   48M     |   9.4G  |   224 |  84.6  | soon |
| iFormer-L  |   87M     |   14.0G |   224 |  84.8  | soon |

Fine-tuning Results with larger resolution (384x384) on ImageNet-1K

| Model      |  #params  | FLOPs | Image resolution | acc@1| Model |
| :---       |   :---:   |  :---: |  :---: |  :---:  |  :---:  |
| iFormer-S  |   20M     |   16.1G  |   384 |  84.6  | soon |
| iFormer-B  |   48M     |   30.5G  |   384 |  85.7  | soon |
| iFormer-L  |   87M     |   45.3G  |   384 |  85.8  | soon |

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

## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).


Besides, Weihao Yu would like to thank TPU Research Cloud (TRC) program for the support of partial computational resources.
