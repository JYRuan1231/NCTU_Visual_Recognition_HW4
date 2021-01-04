from os import listdir
from os.path import join
import numpy
import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size, pad_if_needed=True),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(360),
#         ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // (upscale_factor), interpolation=Image.BILINEAR),
#         Resize(crop_size // (upscale_factor), interpolation=Image.BICUBIC),
        ToTensor(),
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        
        
        cv_hr = cv2.cvtColor(numpy.asarray(hr_image),cv2.COLOR_RGB2BGR)
        
        hr_image = ToTensor()(hr_image)

        lr_image =  cv2.resize(cv_hr, (self.crop_size // (self.upscale_factor),self.crop_size // (self.upscale_factor)), interpolation=cv2.INTER_CUBIC)
        lr_image = Image.fromarray(cv2.cvtColor(lr_image,cv2.COLOR_BGR2RGB)) 
        lr_image = ToTensor()(lr_image)
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.hr_image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_filenames[index]).convert('RGB')
        w, h = hr_image.size
        w_crop_size = calculate_valid_crop_size(w, self.upscale_factor)
        h_crop_size = calculate_valid_crop_size(h, self.upscale_factor)      
        lr_scale = Resize((h_crop_size // self.upscale_factor, w_crop_size// self.upscale_factor), interpolation=Image.BILINEAR)     
        hr_image = Resize((h_crop_size, w_crop_size), interpolation=Image.BICUBIC)(hr_image)
        lr_image = lr_scale(hr_image)
        return ToTensor()(lr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.hr_image_filenames)
    
    
