from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy as np
from mmengine.evaluator import BaseMetric
from torch.optim import AdamW
from EasyMedAI.models.baseModels.base import LvmBaseModel
#Metric def
class IoU(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: str  = "t", collect_dir: str  = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
    def process(self, data_batch, data_samples):
        preds, labels = data_samples[0], data_samples[1]['labels']
        preds = torch.argmax(preds, dim=1)
        intersect = (labels == preds).sum()
        union = (torch.logical_or(preds, labels)).sum()
        iou = (intersect / union).cpu()
        self.results.append(
            dict(batch_size=len(labels), iou=iou * len(labels)))

    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(iou=total_iou / num_samples)
#transforms def
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#网络
class Resnet50_pretrained(LvmBaseModel):

    def __init__(self, num_classes,user_to_head:bool=False):
        super(Resnet50_pretrained, self).__init__(num_classes,user_to_head)
        self.transform_img=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)])
        
        self.transform_lable= transforms.Lambda(
        lambda x: torch.tensor(np.array(x), dtype=torch.long))
        self.optim=dict(type=AdamW, lr=2e-4)
        self.metrics:list[BaseMetric]=[IoU]
        self.user_to_head=user_to_head
        self.resnet50 = deeplabv3_resnet50()
        self.resnet50.classifier[4] = torch.nn.Conv2d(
            256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
    def lossFun(self,x,data_samples):
        return F.cross_entropy(x, data_samples['labels'])
    def forward(self, imgs):
        x = self.resnet50(imgs)['out']
        return x
