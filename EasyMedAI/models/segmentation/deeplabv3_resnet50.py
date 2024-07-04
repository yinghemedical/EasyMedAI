from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy as np
from mmengine.evaluator import BaseMetric
from torch.optim import AdamW
import torchvision.models as models
from EasyMedAI.models.baseModels.base import LvmBaseModel
#Metric def
class IoU(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: str  = "t", collect_dir: str  = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
    def process(self, data_batch, data_samples):
        # eps = 1e-6
        preds, labels = data_samples[0], data_samples[1]['labels']
        # labels=labels[:,-1,:,:]
        preds=F.interpolate(preds,size=(labels.shape[1],labels.shape[2]), mode='bilinear')
        labels=torch.as_tensor(labels,dtype=torch.long)
        preds = torch.argmax(preds, dim=1)
        # intersect = (labels == preds).sum()
        intersection = (preds & labels).float().sum()
        union = (preds | labels).float().sum()
        iou = intersection / union if union != 0 else union
        # iou = (intersection + eps) / (union + eps)
        dice = (2. * intersection) / (preds.sum() + labels.sum() )
        # union = (torch.logical_or(preds, labels)).sum()
        # iou = (intersect / union).cpu()
        self.results.append(
            dict(batch_size=len(labels), iou=iou.mean(),dice=dice.mean() ))

    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        total_dice = sum(result['dice'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(iou=total_iou / num_samples,dice=total_dice/num_samples)
#transforms def
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#网络
class resnet50(LvmBaseModel):
    def __init__(self, num_classes: int, user_to_head: bool = False,pretrained=False):
        super(resnet50, self).__init__(num_classes,user_to_head)
        self.transform_img=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)])
        self.transform_lable= transforms.Lambda(
        lambda x: torch.tensor(np.array(x), dtype=torch.long))
        # transforms.Compose([
        #     transforms.ToTensor()
        # ])
        
        self.optim=dict(type=AdamW, lr=2e-4)
        self.metrics:list[BaseMetric]=[IoU]
        self.user_to_head=user_to_head
        if pretrained:
            self.resnet50 = deeplabv3_resnet50()
        else:
            self.resnet50 = deeplabv3_resnet50(weights=None,weights_backbone=None)
        self.resnet50.classifier[4] = torch.nn.Conv2d(
            256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
    def lossFun(self,x,data_samples):
        # lable= data_samples['labels'][:,-1,:,:]
        lable= torch.as_tensor(data_samples['labels'][:,:,:],dtype=torch.int64)
        x=F.interpolate(x,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
        return F.cross_entropy(x, lable)
    def forward(self, imgs):
        x = self.resnet50(imgs)['out']
        return x
class resnet50_pretrained(resnet50):

    def __init__(self, num_classes,user_to_head:bool=False):
        super(resnet50_pretrained, self).__init__(num_classes,user_to_head,True)
    def lossFun(self,x,data_samples):
        return F.cross_entropy(x, data_samples['labels'])
    def forward(self, imgs):
        x = self.resnet50(imgs)['out']
        return x
    
