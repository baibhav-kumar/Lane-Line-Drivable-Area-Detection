import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

def random_perspective(combination, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    img, gray, line = combination
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2
    
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2
    
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    M = T @ S @ R @ P @ C
    
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)
    
    combination = (img, gray, line)
    return combination

class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset - now with optional root_dir parameter
    '''
    def __init__(self, root_dir='/home/iotlab/iotlab_project/data/bdd100k', transform=None, valid=False):

        '''
        :param root_dir: Root directory for BDD100K dataset
        :param transform: Type of transformation. Default is None.
        :param valid: If True, load validation set, else load training set
        '''
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid = valid
        
        # Set paths based on valid flag
        if valid:
            self.root = os.path.join(root_dir, 'images', 'val')
        else:
            self.root = os.path.join(root_dir, 'images', 'train')
        
        # Check if directory exists
        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"\nDataset directory not found: {self.root}\n"
                f"Please update the root_dir in DataSet.py or create the directory structure:\n"
                f"  {root_dir}/images/train/\n"
                f"  {root_dir}/images/val/\n"
                f"  {root_dir}/segments/train/\n"
                f"  {root_dir}/segments/val/\n"
                f"  {root_dir}/lane/train/\n"
                f"  {root_dir}/lane/val/\n"
            )
        
        self.names = os.listdir(self.root)
        print(f"Loaded {'validation' if valid else 'training'} dataset: {len(self.names)} images from {self.root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_ = 640
        H_ = 360
        
        image_name = os.path.join(self.root, self.names[idx])
        image = cv2.imread(image_name)
        
        # Load segmentation labels
        label1 = cv2.imread(image_name.replace("images", "segments").replace("jpg", "png"), 0)
        label2 = cv2.imread(image_name.replace("images", "lane").replace("jpg", "png"), 0)
        
        # Data augmentation (only for training)
        if not self.valid:
            if random.random() < 0.5:
                combination = (image, label1, label2)
                (image, label1, label2) = random_perspective(
                    combination=combination,
                    degrees=10,
                    translate=0.1,
                    scale=0.25,
                    shear=0.0
                )
            
            if random.random() < 0.5:
                augment_hsv(image)
            
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
        
        # Resize
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        image = cv2.resize(image, (W_, H_))
        
        # Create binary masks
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)
        
        # Convert to tensors
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        
        # Stack for drivable area and lane line
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
        
        # Prepare image
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        
        return image_name, torch.from_numpy(image), (seg_da, seg_ll)
