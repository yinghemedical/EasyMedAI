
from EasyMedAI.models.baseModels.base import LvmBaseModel
import torch
import torchvision.transforms as transforms
from mmengine.evaluator import BaseMetric
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from torch.hub import load
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
class dinov2_base_pretrained(LvmBaseModel):
    def __init__(self, num_classes,user_to_head:bool=True,dino_backbone="dinov2_s"):
        super(dinov2_base_pretrained, self).__init__(num_classes,user_to_head)
        dinoConfig=dino_backbones[dino_backbone]
        self.model = load('facebookresearch/dinov2', dinoConfig["name"])
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
            # x = self.model.get_intermediate_layers(x, n=4, return_class_token=True)
            x = self.model.forward_features(x.cuda())
            # cls_token = x["x_norm_clstoken"]
            # patch_tokens = x["x_norm_patchtokens"]
            # # fmt: off
            # linear_input = torch.cat([
            #     cls_token,
            #     patch_tokens.mean(dim=1),
            # ], dim=1)
            x = x['x_norm_patchtokens']
            # image_like_patches = rows_of_patches.reshape(-1, H, W, C)
            x = x.permute(0,2,1)
            x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))
            
        return x