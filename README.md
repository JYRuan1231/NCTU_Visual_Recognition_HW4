# Visual_Recognition_HW4

## Introduction
The project is training an accurate instance segmentation neteork using Mask R-CNN for [COCO](https://cocodataset.org/#home).

## Usage
We training and testing with Python 3.6, pytorch 1.4 and **Need** to reference [timm](https://github.com/rwightman/pytorch-image-models), [AutoAugment](https://github.com/DeepVoltaire/AutoAugment), [DataAugmentationForObjectDetection](https://github.com/Paperspace/DataAugmentationForObjectDetection), [mAP](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) and [BiFPN](https://github.com/tristandb/EfficientDet-PyTorch).

### Traning and Testing model
First, this Mask R-CNN uses EfficientNet-b7 as a backbone to extract features.
If you want to use this network for training, you must make data fit Coco's specifications.

In addition, **my [model](https://drive.google.com/file/d/19v0EyFfpqsyLoxfYYEr3qUhmZq8U1Hve/view?usp=sharing) can be loaded for testing**, and this is described in detail in the **Testing model** section.


Before training data, Upload the training images to `/data/train_images/` and the test data to `/data/test_images/`.
Make sure you have the coco json file(default name: **pascal_train.json and test.json**) in the `/data/` folder.


### Traning Model

Example:

```
python train.py
```

***important arguments about Traning Model in config.py***

Default:
These arguments is basic setting for model.

| Argument    | Default value |
| ------------|:-------------:|
|model_name             |  mask_rcnn_v1             |
|batch_size             |  4            |
|accumulation_steps             |  4   (Gradient accumulation)         |
|workers             |  4             |
|num_classes             |  21  (classes+background)           |
|max_size             |  680  (Resize image )           |
|min_size             |  680  (Resize image )           |

And, these is related to your training performance.

| Argument    | Default value |
| ------------|:-------------:|
|epochs             |  60             |
|learning_rate             |  0.005             |
|momentum           |  0.9           |
|num_classes             |  21  (classes+background)           |
|weight_decay             |  5e-4            |
|T_mult             |  1 (CosineAnnealingWarmRestarts)           |
|eta_min             |  0.00001           |

When the program was finished, we will get a traning model file in `/model/`.

```
./models/mask_rcnn_v1
```

### Testing Model

If we want to test image, make sure we have a model in `/models/` and confirm `model_name`.
In addition, if you want to use my trained model, download this [model](https://drive.google.com/file/d/19v0EyFfpqsyLoxfYYEr3qUhmZq8U1Hve/view?usp=sharing) to `/models/` and change the "model_name" to `mask_rcnn_effb7` in `config.py`

Example:

```
python test.py
```

***important arguments about Testing Model in config.py***


Default:

| Argument    | Default value |
| ------------|:-------------:|
|json_name             |      0856566.json       |

When the program was finished, we will get a json file in /result/.
```
./result/0856566.json
```
## Optional
If you want to replace the FPN to BiFPN, or evaluate the AP of training model, please use the following parameters to enable these function.

Default:

| Argument    | Default value |
| ------------|:-------------:|
|bifpn             |      False       |
|eval_train             |      False       |

I strongly suggest not to enable BiFPN because the performance of my model which adopts BiFPN (mAP:0.45) is lower than FPN (mAP:0.65)


## Result

| Metrics    | value |
| ------------|:-------------:|
|mAP             |     <img src="image/mAP.png" width=200>          |
