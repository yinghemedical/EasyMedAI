from EasyMedAI.models.baseModels.base import LvmBaseModel
import torch
import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from torch.hub import load
import torch.nn as nn

class IoU(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: str  = "t", collect_dir: str  = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
    def process(self, data_batch, data_samples):
        preds, labels = data_samples[0], data_samples[1]['labels']
        preds=F.interpolate(preds,size=(labels.shape[2],labels.shape[3]), mode='bilinear')
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
class conv_s(LvmBaseModel):
    def __init__(self, num_classes,user_to_head:bool=True):
        super(conv_s, self).__init__(num_classes,user_to_head)
        
        self.model=conv_base_net(embedding_size=384,num_classes=num_classes,user_to_head=user_to_head)
        self.transform_img=self.model.transform_img
        self.transform_lable=self.model.transform_lable
        self.optim=self.model.optim
        self.metrics=self.model.metrics
        self.user_to_head=self.model.user_to_head
        self.lossFun=self.model.lossFun
    def forward(self, x):
        return self.model(x)
class conv_base_net(LvmBaseModel):
    def __init__(self, embedding_size = 384, num_classes = 5,user_to_head=False):
        super(conv_base_net, self).__init__(num_classes,user_to_head=True)
        self.transform_img= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((14*32,14*32)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_lable= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32*4,32*4)),
        ])
        self.optim=dict(type=AdamW, lr=0.0001)
        self.metrics:list[BaseMetric]=[IoU]
        self.user_to_head=user_to_head
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(3, 3, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(3, num_classes, (3,3), padding=(1,1)),
            # nn.MaxPool2d((4,4),stride=4,padding=(1,1))
        )
    def lossFun(self,x,data_samples):
        task_type=data_samples["task_type"][0]
        if task_type != "Segmentation":
            raise RuntimeError("not support "+task_type)
        lable= torch.as_tensor(data_samples['labels'][:,-1,:,:],dtype=torch.int64)
        x=F.interpolate(x,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
        # lable=F.upsample(lable,(x.shape[2],x.shape[3]))
        # _, predicted = torch.max(x, 1)
        # predicted=predicted.to(torch.float16)[:,None,:,:]
        return F.cross_entropy(x, lable)
    def forward(self, x):
        # szie=(x.shape(2),x.shape(3))
        x = self.model(x)
        # x = x.to(torch.float32)
        x = torch.sigmoid(x)
        
        # _, predicted = torch.max(x, 1)
        return x