from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numpy as np
import os
class segmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, images=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        if images is None:
            self.images=[]
            for  img in  os.listdir(img_dir):
                maskFile=os.path.join(mask_dir, img)
                if os.path.isfile(maskFile):
                    self.images.append(img)
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        bin_mask = torch.zeros(self.num_classes, mask.shape[1], mask.shape[2])
        mask = torch.from_numpy(np.array(mask)).to(bin_mask.device)
        #num_classes 是类型的数量+1，会自动处理0为背景
        for i in range(self.num_classes):
            bin_mask[i] = (mask == i).float() 
        return image, bin_mask