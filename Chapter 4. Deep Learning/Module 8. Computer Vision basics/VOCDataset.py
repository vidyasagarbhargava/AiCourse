import torch
from torchvision import transforms
import os
from PIL import Image

label_transform = transforms.Compose([
    transforms.ToTensor()
])

class VOC(torch.utils.data.Dataset):
    def __init__(self, input_transform=None):
        self.filenames = [fn.split('.')[0] for fn in os.listdir('./VOC2007/SegmentationClass')]
        self.input_transform = input_transform

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(f'./VOC2007/JPEGImages/{fn}.jpg')
        label = Image.open(f'./VOC2007/SegmentationClass/{fn}.png')

        if self.input_transform:
            img = self.input_transform(img)

        label = label_transform(label)
        return (img, label)
    
    def __len__(self):
        return len(self.input_fps)
