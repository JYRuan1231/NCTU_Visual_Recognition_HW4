# Visual_Recognition_HW4

## Introduction
The project is training an neteork using SRGAN for Super-Resolution task.

## Usage
We training and testing with Python 3.6, pytorch 1.4 and **reference** [SRGAN](https://github.com/leftthomas/SRGAN)

### Traning and Testing model
Before training data, Upload the training images to `/data/training_hr_images/` and the test data to `/data/testing_lr_images/`.

Make sure split some training data to `/data/validation/` as validation dataset.

### Directory Structure
```
project
│   README.md
│   config.py
|   data_utils.py
|   loss_srgan.py
│   model_srgan.py
|   test_srgan.py
|   train_srgan.py
│
└───pytorch_ssim
│   │   __init__.py
│   
└───images
|   |   00.png
└───data
|    │───training_hr_images
|    │───testing_lr_images
|    │───validation
└───saved_models
```


### Traning Model

Example:

```
python train_srgan.py
```

***important arguments about Traning Model in config.py***

Default:
These arguments is basic setting for model.

| Argument    | Default value |
| ------------|:-------------:|
|model_name             |  generator.pth             |
|batch_size             |  32            |
|workers             |  4             |
|crop_size             |  48           |

And, these is related to your training performance.

| Argument    | Default value |
| ------------|:-------------:|
|epochs             |  30000             |
|adam_lr             |  5e-4             |
|beta1(for adam)             |  0.5             |
|sgd_lr           |  1e-4           |

The following are for CosineAnnealingWarmRestar

| Argument    | Default value |
| ------------|:-------------:|
|T_0           |  5000      |
|T_mult             |  2            |
|sgd_eta_min             |  1e-5           |
|adam_eta_min             |  5e-5           |

When the program was finished, we will get a traning model file in `/saved_models/`.

```
./saved_models/generator.pth
```

### Testing Model

If we want to test image, make sure we have a model in `/saved_models/` and confirm `model_name`.
In addition, if you want to use my trained model, download this [model](https://drive.google.com/file/d/18-UqHy4TOCS2HLfPm4bUhtrBat-x-qD9/view?usp=sharing) to `/saved_models/` and change the "model_name" to `srgan_v1.pth` in `config.py`

Example:

```
python test_srgan.py
```

***important arguments about Testing Model in config.py***


Default:

| Argument    | Default value |
| ------------|:-------------:|
|model_name             |      generator.pth       |

When the program was finished, we will get 14 high-resolution images file in /images/.
```
./images/00.png
./images/01.png
.
.
.
./images/13.png
```

