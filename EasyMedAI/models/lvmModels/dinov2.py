
import itertools
import math
from EasyMedAI.models.baseModels.base import LvmBaseModel
import torch
import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from torch.hub import load
import torch.nn as nn
import warnings
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
dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}

class ConvertOutput(nn.Module):
    """Just a batchnorm."""

    def __init__(self,channels,input_transform="resize_concat",resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        # self.in_channels == channels
        self.bn = nn.SyncBatchNorm(channels)
        self.resize_factors = resize_factors
        self.input_transform=input_transform
        self.in_index=[3]
        self.align_corners=False
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
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
            inputs = [inputs[i] for i in self.in_index]
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
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=True)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        # output = self.cls_seg(output)
        return output
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
        self.model = load('facebookresearch/dinov2', dinoConfig["name"])
        if hasattr(self.model, "patch_size"):
            self.model.register_forward_pre_hook(lambda _, x: CenterPadding(self.model.patch_size)(x[0]))
        # t=self.model.get_intermediate_layers()
        self.transform_img= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((14*32,14*32)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_lable= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32*4,32*4)),
        ])
        self.optim=dict(type=Adam, lr=2e-4)
        self.metrics:list[BaseMetric]=[IoU]
        self.user_to_head=user_to_head
        self.patch_size=dinoConfig["patch_size"]
        self.embedding_size=dinoConfig["embedding_size"]
        self.convertOutput=ConvertOutput(self.embedding_size,resize_factors=[4])
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
    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        with torch.no_grad():
            embedding = self.model.get_intermediate_layers(x, n=[8, 9, 10, 11], reshape=True)
            output =self.convertOutput(embedding)

            # output = self.model.forward_features(x.cuda())
            # # cls_token = x["x_norm_clstoken"]
            # # patch_tokens = x["x_norm_patchtokens"]
            # # # fmt: off
            # # linear_input = torch.cat([
            # #     cls_token,
            # #     patch_tokens.mean(dim=1),
            # # ], dim=1)
            # output = output['x_norm_patchtokens']
            # # upsample = nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
            # # t=upsample(output.unsqueeze(1))
            # output = output.permute(0,2,1)
            # # feature_map = output.reshape(32, -1, 7, 7)
            # output = output.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))
           
            # image_like_patches = rows_of_patches.reshape(-1, H, W, C)
            
            
            
        return output