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
#https://github.com/shuxinyin/NLP-Loss-Pytorch/blob/master/unbalanced_loss/dice_loss.py
def calculateBinaryMetric(predict,target,smooth=1,p=2):
    # pred = torch.sigmoid(pred)
    # ft= pred>0.5
    # pred =pred*ft
    # predict = torch.sigmoid(predict)
    # ft= predict>0.5
    # predict =predict*ft
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

    dice= (num / den).mean()
    return dice

class binary_metric(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: str  = "t", collect_dir: str  = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
    def process(self, data_batch, data_samples):
        # eps = 1e-6
        preds, labels = data_samples[0], data_samples[1]['labels']
        
        iou,dice=calculateBinaryMetric(preds,labels)
        self.results.append(
            dict(batch_size=len(labels), iou=iou.mean(),dice=dice.mean() ))

    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        total_dice = sum(result['dice'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(iou=total_iou / num_samples,dice=total_dice/num_samples)
class IoU(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: str  = "t", collect_dir: str  = None) -> None:
        super().__init__(collect_device, prefix, collect_dir)
    def process(self, data_batch, data_samples):
        # eps = 1e-6
        predict, lable = data_samples[0], data_samples[1]['labels']
        total_dice = 0
        predict=F.interpolate(predict,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
        predict = F.softmax(predict, dim=1)
        for i in range(predict.shape[1]):
            if i>0:
                pr=predict[:,i,:,:]
                la = lable==i
                dice=calculateBinaryMetric(pr,la)
                
                total_dice += dice
        dice= total_dice/(predict.shape[1]-1)
        # labels=labels[:,-1,:,:]
        # preds=F.interpolate(preds,size=(labels.shape[1],labels.shape[2]), mode='bilinear')
        # labels=torch.as_tensor(labels,dtype=torch.long)
        # preds = torch.argmax(preds, dim=1)
        # # preds = preds[:,1,:]
        # # preds = torch.argmax(preds,keepdim=True)
        # # intersect = (labels == preds).sum()
        # intersection = (preds & labels).float().sum()
        # union = (preds | labels).float().sum()
        # iou = intersection / union if union != 0 else union
        # # iou = (intersection + eps) / (union + eps)
        # dice = (2. * intersection) / (preds.sum() + labels.sum() )
        # # union = (torch.logical_or(preds, labels)).sum()
        # # iou = (intersect / union).cpu()
        self.results.append(
            dict(batch_size=len(lable),dice=dice ))

    def compute_metrics(self, results):
        # total_iou = sum(result['iou'] for result in self.results)
        total_dice = sum(result['dice'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(idice=total_dice/num_samples)
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
            self.resnet50 = deeplabv3_resnet50(num_classes=num_classes)
        else:
            self.resnet50 = deeplabv3_resnet50(num_classes=num_classes,weights=None,weights_backbone=None)
        self.resnet50.classifier[4] = torch.nn.Conv2d(
            256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
    def binary_dice_coefficient(self,pred, target, smooth=1.):
        # 预测结果 pred 需要通过 Sigmoid 函数转换为概率
        pred = torch.sigmoid(pred)
        ft= pred>0.5
        pred =pred*ft
        # 将预测结果和目标标签转换为相同形状的一维向量
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        total = pred_flat.sum() + target_flat.sum()
        
        # 计算 Dice 系数
        dice_coeff = (2. * intersection + smooth) / (total + smooth)
        
        return dice_coeff
    def BinaryDSCLoss(self, probs, targets,alpha=1.0,smooth=1.0):
        probs = torch.sigmoid(probs)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        targets = targets.unsqueeze(dim=1)
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        pos_weight = pos_mask * ((1 - probs) ** alpha) * probs
        pos_loss = 1 - (2 * pos_weight + smooth) / (pos_weight + 1 + smooth)

        neg_weight = neg_mask * ((1 - probs) ** alpha) * probs
        neg_loss = 1 - (2 * neg_weight + smooth) / (neg_weight + smooth)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss
    def lossFun(self,x,data_samples):
        # loss=self.dice_loss(x,data_samples['labels'])
        # return loss
        if self.num_classes >2:
            lable= torch.as_tensor(data_samples['labels'][:,:,:],dtype=torch.float)
            x=F.interpolate(x,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
            return F.cross_entropy(x, lable)
        else:
            lable= data_samples['labels'][:,:,:]
            # lable= torch.as_tensor(data_samples['labels'][:,:,:],dtype=torch.float)
            # x=F.interpolate(x,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
            # x=x[:,1,:,:]
            # # x=x1+x[:,0,:,:]
            # x=torch.as_tensor(x,dtype=torch.float)
            # loss=1-self.binary_dice_coefficient(x,lable)
            total_loss = 0
            x=F.interpolate(x,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
            lable=data_samples['labels'][:,:,:]
            predict = F.softmax(x, dim=1)
            for i in range(predict.shape[1]):
                if i>0:
                    pr=predict[:,i,:,:]
                    la = lable==i
                    dice=calculateBinaryMetric(pr,la)
                    dice_loss =1-dice
                    total_loss += dice_loss
            return total_loss/(predict.shape[1]-1)
            # x=F.interpolate(x,size=(lable.shape[1],lable.shape[2]), mode='bilinear')
            # preds = torch.argmax(x, dim=1)
            # x=x[:,1,:,:]
            # dice=calculateBinaryMetric(x,lable)
            # loss= 1- dice
            # return loss
    def forward(self, imgs):
        x = self.resnet50(imgs)['out']
        pred = torch.sigmoid(x)
        ft= pred>0.5
        pred = pred*ft
        return x
class resnet50_pretrained(resnet50):

    def __init__(self, num_classes,user_to_head:bool=False):
        super(resnet50_pretrained, self).__init__(num_classes,user_to_head,True)
    def lossFun(self,x,data_samples):
        return F.cross_entropy(x, data_samples['labels'])
    def forward(self, imgs):
        x = self.resnet50(imgs)['out']
        return x
    
