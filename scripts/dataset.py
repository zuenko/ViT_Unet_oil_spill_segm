"""
Dataset classes for oil spill segmentation
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OilSpillDataset(Dataset):
    """
    Base dataset for oil spill segmentation.
    
    Args:
        images_dir: Directory containing input images
        masks_dir: Directory containing ground truth masks
        transform_image: Albumentations transform for images
        transform_mask: Albumentations transform for masks
        sensor_type: Optional sensor type ('palsar' or 'sentinel')
    """
    def __init__(self, images_dir, masks_dir, transform_image=None, 
                 transform_mask=None, sensor_type=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        if os.path.exists(images_dir):
            self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.image_files = []
            self.mask_files = []
            
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.sensor_type = sensor_type
        
        if len(self.image_files) != len(self.mask_files):
             # Try matching by name if counts differ (sometimes happens in orig datasets)
             common = sorted(list(set(self.image_files) & set(self.mask_files)))
             self.image_files = common
             self.mask_files = common
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        if self.transform_image:
            augmented = self.transform_image(image=image)
            image = augmented["image"]
        
        if self.transform_mask:
            augmented = self.transform_mask(image=mask)
            mask = augmented["image"]
        
        mask = (mask > 0).float()
        
        return image, mask


def load_dataset_orig(root_dir, split='train', transform_img=None, transform_mask=None):
    """Loads dataset_orig structure by merging palsar and sentinel."""
    palsar_img = os.path.join(root_dir, split, 'palsar', 'image')
    palsar_mask = os.path.join(root_dir, split, 'palsar', 'label')
    
    sentinel_img = os.path.join(root_dir, split, 'sentinel', 'image')
    sentinel_mask = os.path.join(root_dir, split, 'sentinel', 'label')
    
    datasets = []
    if os.path.exists(palsar_img):
        datasets.append(OilSpillDataset(palsar_img, palsar_mask, transform_img, transform_mask, 'palsar'))
    if os.path.exists(sentinel_img):
        datasets.append(OilSpillDataset(sentinel_img, sentinel_mask, transform_img, transform_mask, 'sentinel'))
        
    if not datasets:
        return None
    return ConcatDataset(datasets)


def get_transforms(image_size=(224, 224)):
    """
    Get training and validation transforms.
    """
    transform_image_val = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    transform_mask = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        ToTensorV2(),
    ])
    
    transform_image_train = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return transform_image_train, transform_mask, transform_image_val, transform_mask
