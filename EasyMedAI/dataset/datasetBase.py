import os
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
import csv
import torch
from EasyMedAI.enums import DataSetLoadType, TaskType

def create_color_to_class(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class

class datasetBase(VisionDataset):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 load_type= DataSetLoadType.Png,
                 task_type =TaskType.Segmentation,
                 ):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = None
        self.mask_folder = None
        self.images = None
        self.masks = None
        self.masks = None
        self.load_type=load_type
        self.task_type=task_type
        # self.color_to_class = create_color_to_class(class_list)
    def getMasks(self,index):
        raise NotImplementedError()
    def getClass(self,index):
        raise NotImplementedError()
    def __getitem__(self, index):

        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder,
                                 self.masks[index])
        if self.load_type==DataSetLoadType.Png:
            img = Image.open(img_path).convert('RGB')
            # mask = Image.open(mask_path).convert('RGB')  # Convert to RGB
            if self.transform is not None:
                img = self.transform(img)
            if self.load_type==TaskType.Segmentation:
                # Convert the RGB values to class indices
                # mask = np.array(mask)
                # mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
                # labels = np.zeros_like(mask, dtype=np.int64)
                # for color, class_index in self.color_to_class.items():
                #     rgb = color[0] * 65536 + color[1] * 256 + color[2]
                #     labels[mask == rgb] = class_index
                labels =self.getMasks(index)
                if self.target_transform is not None:
                    labels = self.target_transform(labels)
                classes=[]
            elif self.load_type==TaskType.Classification:
                labels =[]
                classes =self.getClass(index)
                classes = torch.as_tensor(classes)
            data_samples = dict(
                labels=labels, img_path=img_path, mask_path=mask_path, classes=classes,task_type=self.load_type)
            return img, data_samples

    def __len__(self):
        return len(self.images)