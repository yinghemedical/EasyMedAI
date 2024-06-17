import os
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
# class MedMNIST2D(VisionDataset):
#      def __init__(self,
#                  root,
#                  folder,
#                  transform=None,
#                  target_transform=None):
#         super().__init__(
#             root, transform=transform, target_transform=target_transform)
#         self.folder = folder

#         self.mask_folder = mask_folder
#         self.images = list(
#             sorted(os.listdir(os.path.join(self.root, img_folder))))
#         self.masks = list(
#             sorted(os.listdir(os.path.join(self.root, mask_folder))))
#         self.color_to_class = create_palette(
#             os.path.join(self.root, 'class_dict.csv'))

#     def __getitem__(self, index):
#         img_path = os.path.join(self.root, self.img_folder, self.images[index])
#         mask_path = os.path.join(self.root, self.mask_folder,
#                                  self.masks[index])

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(mask_path).convert('RGB')  # Convert to RGB

#         if self.transform is not None:
#             img = self.transform(img)

#         # Convert the RGB values to class indices
#         mask = np.array(mask)
#         mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
#         labels = np.zeros_like(mask, dtype=np.int64)
#         for color, class_index in self.color_to_class.items():
#             rgb = color[0] * 65536 + color[1] * 256 + color[2]
#             labels[mask == rgb] = class_index

#         if self.target_transform is not None:
#             labels = self.target_transform(labels)
#         data_samples = dict(
#             labels=labels, img_path=img_path, mask_path=mask_path)
#         return img, data_samples

#     def __len__(self):
#         return len(self.images)