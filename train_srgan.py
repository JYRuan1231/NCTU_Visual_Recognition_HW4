import argparse
import os
from math import log10
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from torch.optim import lr_scheduler
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder
from loss_srgan import GeneratorLoss
from model_srgan import Generator, Discriminator
import skimage.measure
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import cv2
import math


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    # print(sr.size())
    # print(hr.size())
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    # shave = int(shave)
    import math
    shave = math.ceil(shave)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)



class Opt:
    def __init__(self, crop_size=48, num_epochs=30000, upscale_factor=3):
        self.crop_size = crop_size
        self.num_epochs = num_epochs
        self.upscale_factor = upscale_factor


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = Opt()
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = TrainDatasetFromFolder('data/training_hr_images', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/validation', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=5e-4, betas=(0.5, 0.999))
    schedulerG = lr_scheduler.CosineAnnealingWarmRestarts(optimizerG, T_0=5000, T_mult=2, eta_min=5e-5)

    optimizerD = optim.SGD(netD.parameters(), lr=1e-4)
    schedulerD = lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0=5000, T_mult=2, eta_min=1e-5)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    best_psnr = 0.0
    best_ssim = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        since = time.time()
        total_lossD = 0.0
        total_lossG = 0.0

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            optimizerD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()

            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            optimizerG.zero_grad()

            g_loss = generator_criterion(netD(fake_img).mean(), netG(z), real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()
            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr2': 0, 'ssim': 0, 'batch_sizes': 0}

            for iter, (val_lr, val_hr) in enumerate(val_bar):
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()

                if hr.shape[2] / lr.shape[2] != 3 or hr.shape[3] / lr.shape[3] != 3:
                    print(hr.shape, lr.shape)

                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size

                max_num = 1.0
                sr = quantize(sr, max_num)
                hr = quantize(hr, max_num)

                valing_results['psnr2'] += calc_psnr(sr, hr, 3, max_num) / 14


                if best_psnr < valing_results['psnr2']:
                    best_psnr = valing_results['psnr2']
                    # Save model checkpoints
                    torch.save(netG.state_dict(), "saved_models/generator.pth")
                    print('save best psnr success!')

                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR2: %.4f dB SSIM: %.4f' % (
                     valing_results['psnr2'], valing_results['ssim']))

        schedulerD.step()
        schedulerG.step()
        time_elapsed = time.time() - since
        print('Complete one epoch in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))