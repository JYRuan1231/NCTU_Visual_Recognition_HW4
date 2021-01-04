import argparse
import os
from math import log10
from PIL import Image
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms 
import skimage.measure
from model_srgan import Generator, Discriminator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Opt:
    def __init__(self, crop_size=78, num_epochs=50, upscale_factor=3):
        self.crop_size = crop_size
        self.num_epochs = num_epochs
        self.upscale_factor = upscale_factor

opt = Opt()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs


test_path = './data/testing_lr_images/'
allFileList = os.listdir(test_path)
model = Generator(UPSCALE_FACTOR)
model.load_state_dict(torch.load('./saved_models/'+'generator.pth'))

for file in allFileList:
    if os.path.isfile(test_path + file):
        path = test_path + file
        img = Image.open(path)
        img = Variable(ToTensor()(img)).unsqueeze(0)
        with torch.no_grad():
            model.eval().to(device)
            img = img.to(device)
            outputs = model(img)
            out_img = ToPILImage()(outputs[0].data.cpu())
            out_img.save('./result/'+ file)
            plt.imshow(out_img)
            plt.show()
  

        


