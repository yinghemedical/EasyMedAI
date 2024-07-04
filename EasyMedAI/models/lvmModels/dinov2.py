
import itertools
import math

import numpy as np
from EasyMedAI.models.baseModels.base import LvmBaseModel
import torch
import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from torch.hub import load
import torch.nn as nn
import warnings
import os
# torch.hub.DEFAULT_CACHE_DIR
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.apis import init_segmentor, inference_segmentor
# from mmcv.utils import Registry
from EasyMedAI.models.lvmModels.mmops import resize
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
checkpoint_root=os.environ.get('easymedai_pretrained_models')
if checkpoint_root == None:
    checkpoint_root='./easymedai_pretrained_models/dinov2'
else:
    checkpoint_root=checkpoint_root+"/dinov2"
dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14,
        'output_size':32,
        'checkpoint_name':'dinov2_vits14_pretrain.pth',
        'checkpoint_downloadurl':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14,
        'output_size':32,
        'checkpoint_name':'dinov2_vitb14_pretrain.pth',
        'checkpoint_downloadurl':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14,
        'output_size':32,
        'checkpoint_name':'dinov2_vitl14_pretrain.pth',
        'checkpoint_downloadurl':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14,
        'output_size':32,
        'checkpoint_name':'dinov2_vitg14_pretrain.pth',
        'checkpoint_downloadurl':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'
    },
}


class dinov2_s_pretrained(LvmBaseModel):
    def __init__(self, num_classes,user_to_head:bool=True):
        super(dinov2_s_pretrained, self).__init__(num_classes,user_to_head)
        dino_backbone="dinov2_s"
        self.model=dinov2_base_pretrained(num_classes,user_to_head,dino_backbone=dino_backbone)

        self.transform_img=self.model.transform_img
        self.transform_lable=self.model.transform_lable
        self.optim=self.model.optim
        self.metrics=self.model.metrics
        self.user_to_head=self.model.user_to_head
        self.lossFun=self.model.lossFun
    def forward(self, x):
        return self.model(x)
class dinov2_b_pretrained(LvmBaseModel):
    def __init__(self, num_classes,user_to_head:bool=True):
        super(dinov2_b_pretrained, self).__init__(num_classes,user_to_head)
        dino_backbone="dinov2_b"
        self.model=dinov2_base_pretrained(num_classes,user_to_head,dino_backbone=dino_backbone)

        self.transform_img=self.model.transform_img
        self.transform_lable=self.model.transform_lable
        self.optim=self.model.optim
        self.metrics=self.model.metrics
        self.user_to_head=self.model.user_to_head
        self.lossFun=self.model.lossFun
    def forward(self, x):
        return self.model(x)
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

class dinov2_base_pretrained(LvmBaseModel):
    def __init__(self, num_classes,user_to_head:bool=True,dino_backbone="dinov2_s"):
        super(dinov2_base_pretrained, self).__init__(num_classes,user_to_head)
        dinoConfig=dino_backbones[dino_backbone]
        checkpoint_path= os.path.join(checkpoint_root,dinoConfig["checkpoint_name"]) 
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_root,exist_ok=True)
        #     raise RuntimeError(
        #     f"""
        #     Automatic download failed! Please download dinov2 pretrained model manually.
        #     1. Go to {dinoConfig["checkpoint_downloadurl"]}  Download file and put under your dinov2 model root folder: 
        #     {checkpoint_root} 
        #     """
        #     )

        self.model = load('facebookresearch/dinov2', dinoConfig["name"])
        if hasattr(self.model, "patch_size"):
            self.model.register_forward_pre_hook(lambda _, x: CenterPadding(self.model.patch_size)(x[0]))
        self.patch_size=dinoConfig["patch_size"]
        self.embedding_size=dinoConfig["embedding_size"]
        self.output_size=dinoConfig["output_size"]
        # t=self.model.get_intermediate_layers()
        self.transform_img= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.output_size*self.patch_size,self.output_size*self.patch_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_lable= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.output_size*self.patch_size,self.output_size*self.patch_size)),
        ])
        # self.transform_lable= transforms.Lambda(
        # lambda x: torch.tensor(np.array(x), dtype=torch.long))
        self.optim=dict(type=Adam, lr=2e-4)
        self.metrics:list[BaseMetric]=[IoU]
        self.user_to_head=user_to_head
        self.resize_factors=[self.patch_size]
        self.output_transform="resize_concat"
        self.out_index=[3]
        self.align_corners=False
        # self.convertOutput=ConvertOutput(self.embedding_size,resize_factors=[self.patch_size])
        self.fc=nn.Sequential(nn.Conv2d(self.embedding_size, 3, (2,2), padding=(1,1)))
        self.bn = nn.SyncBatchNorm(self.embedding_size)
        self.up= nn.Upsample(size=(self.patch_size*self.output_size,self.patch_size*self.output_size))
        self.layers=1
    def lossFun(self,x,data_samples):
        return F.cross_entropy(x, data_samples['labels'])
    """
    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)
    """
    def _transform_outputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.output_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.out_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.output_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    def forward(self, x):
        # batch_size = x.shape[0]
        # mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        B, _, w, h = x.shape
        with torch.no_grad():
            # patch_tokens = self.model.forward_features(x)
            # patch_tokens = patch_tokens["x_norm_patchtokens"]
            # # patch_tokens= patch_tokens.permute(0, 2, 1)
            # patch_tokens=patch_tokens.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
               
            outputs = self.model.get_intermediate_layers(x, n=[8, 9, 10, 11], reshape=True)
        # print(embedding)
        outputs =self._transform_outputs(outputs)
        outputs =self.bn(outputs)
        # patch_tokens =self.up(patch_tokens)
        output = self.fc(outputs)
    
            # if self.layers == 1:
            #     x = self.model.forward_features(x)
            #     cls_token = x["x_norm_clstoken"]
            #     patch_tokens = x["x_norm_patchtokens"]
                
            #     output = torch.cat([
            #         cls_token,
            #         patch_tokens.mean(dim=1),
            #     ], dim=1)
            # elif self.layers == 4:
            #     x = self.model.get_intermediate_layers(x, n=4, return_class_token=True)
            #     output = torch.cat([
            #         x[0][1],
            #         x[1][1],
            #         x[2][1],
            #         x[3][1],
            #         x[3][0].mean(dim=1),
            #     ], dim=1)
        # c_1=torch.min(output, dim=(1))[0][:,None,:,:]
        # c_2=torch.mean(output, dim=(1))[:,None,:,:]    
        # c_3=torch.max(output, dim=(1))[0][:,None,:,:]    
        # output=torch.cat((c_1,c_2,c_3),dim=1)
        return output