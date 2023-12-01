import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
import math 
from PIL import Image

class ChairForVan(Dataset):
    def __init__(self,img_dir, image_names, transform=ToTensor(), target_transform=None):
        self.img_dir = img_dir
        self.images_path = [self.img_dir + "/" + i for i in image_names]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image,(224,224))
        label = torch.tensor(20,dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def make_train_val(img_dir,split=0.1,batch_size=1,val_only=False):
    total_data = os.listdir(img_dir)   
    train_data = list()
    val_data = list()
    if not val_only:
        split_val = int(math.floor(len(total_data)*split))
        val_data = random.sample(total_data,split_val)
    for i in total_data:
        if i not in val_data:
            train_data.append(i)
    print(f"Train data : {len(train_data)}, Val data: {len(val_data)}")
    train_data = ChairForVan(img_dir, image_names=train_data)
    train_dataloader =  DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = None
    if not val_only:
        val_data = ChairForVan(img_dir, image_names=val_data)
        val_dataloader =  DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader