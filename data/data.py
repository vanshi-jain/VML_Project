import os
import math
import random
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

class ChairDataset(Dataset):
    """Custom dataset class for loading chair images with a fixed label."""

    def __init__(self, img_dir, image_names, transform=ToTensor(), target_transform=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            image_names (list): List of image filenames.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on the label.
        """
        self.img_dir = img_dir
        self.image_paths = [os.path.join(img_dir, name) for name in image_names]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        label = torch.tensor(20, dtype=torch.int64)  # All samples have the same label

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def make_train_val_dataloaders(img_dir, split=0.1, batch_size=1, val_only=False):
    """
    Creates train and validation dataloaders from a directory of images.

    Args:
        img_dir (str): Path to the image directory.
        split (float): Fraction of data to use for validation.
        batch_size (int): Batch size for the dataloaders.
        val_only (bool): If True, return only the validation dataloader.

    Returns:
        Tuple[DataLoader, DataLoader or None]: Train and validation dataloaders.
    """
    all_images = os.listdir(img_dir)
    val_data = []
    train_data = []

    if not val_only:
        val_size = int(math.floor(len(all_images) * split))
        val_data = random.sample(all_images, val_size)
        train_data = [img for img in all_images if img not in val_data]
    else:
        train_data = []

    print(f"Train data: {len(train_data)}, Validation data: {len(val_data)}")

    train_loader = None
    if not val_only:
        train_dataset = ChairDataset(img_dir, image_names=train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ChairDataset(img_dir, image_names=val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
