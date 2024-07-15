import os
import cv2
import numpy as np
import torch
from EasyMedAI.models.baseModels.base import LvmBaseModel
from EasyMedAI.models.lvmModels.SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
from EasyMedAI.models.lvmModels.mmops import resize
from .segment_anything.build_sam import _build_sam
from .segment_anything.modeling.common import LayerNorm2d
from .segment_anything.utils.transforms import ResizeLongestSide
from .segment_anything.modeling.sam import Sam
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from typing import List, Tuple, Type
from .segment_anything import build_sam_vit_b,build_sam_vit_l,build_sam_vit_h
sam_configs={
    "vit_b":{
        "encoder_embed_dim":768,
        "encoder_depth":12,
        "encoder_num_heads":12,
        "encoder_global_attn_indexes":[2, 5, 8, 11],
        "checkpoint":"sam-med2d_b.pth",
        "download_url":"https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view"
    },
    "vit_l":{
        "encoder_embed_dim":1024,
        "encoder_depth":24,
        "encoder_num_heads":16,
        "encoder_global_attn_indexes":[5, 11, 17, 23],
        "checkpoint":"sam_vit_l_0b3195.pth",
        "download_url":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "vit_h":{
        "encoder_embed_dim":1280,
        "encoder_depth":32,
        "encoder_num_heads":16,
        "encoder_global_attn_indexes":[7, 15, 23, 31],
        "checkpoint":"sam_vit_h_4b8939.pth",
        "download_url":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
}
checkpoint_root=os.environ.get('easymedai_pretrained_models')
if checkpoint_root == None:
    checkpoint_root='./easymedai_pretrained_models/SAM_Med2D'
else:
    checkpoint_root=checkpoint_root+"/SAM_Med2D"
class sam_med2d_base(LvmBaseModel):
    def __init__(self,model_name, num_classes,user_to_head:bool=True):
        super(sam_med2d_base, self).__init__(num_classes,user_to_head)
        self.image_size=256
        if user_to_head:
            raise RuntimeError(
            f"""
            SAM model not support in head
            """
            )
        model_config=sam_configs[model_name]
        checkpoint_path= os.path.join(checkpoint_root,model_config["checkpoint"]) 
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_root,exist_ok=True)
            raise RuntimeError(
            f"""
            Automatic download failed! Please download sam pretrained model manually.
            1. Go to {model_config["download_url"]}  Download file '{model_config["checkpoint"]}' and put under your sam model root folder: 
            {checkpoint_root} 
            """
            )
       
        self.transform_img=  transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((self.image_size,self.image_size)),
        ])
        self.transform_lable= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size,self.image_size)),
        ])
        self.model:Sam=_build_sam(
        encoder_embed_dim=model_config["encoder_embed_dim"],
        encoder_depth=model_config["encoder_depth"],
        encoder_num_heads=model_config["encoder_num_heads"],
        encoder_global_attn_indexes=model_config["encoder_global_attn_indexes"],
        image_size=self.image_size,
        checkpoint=checkpoint_path,
        encoder_adapter=False
    )
        # build_sam_vit_b(checkpoint=checkpoint_path)
        self.transform= self.create_transforms((self.model.image_encoder.img_size,self.model.image_encoder.img_size))
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(self.model.image_encoder.out_chans, self.model.image_encoder.out_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.model.image_encoder.out_chans // 4),
            nn.GELU(),
            nn.ConvTranspose2d(self.model.image_encoder.out_chans // 4, self.model.image_encoder.out_chans // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(self.model.image_encoder.out_chans // 8, 3,(3,3), padding=(1,1)),
            # LayerNorm2d(self.model.image_encoder.out_chans // 4),
            # nn.MaxPool2d((3,3)),
            # nn.Conv2d(self.model.image_encoder.out_chans // 4,3,(3,3), padding=(1,1)),
            # nn.MaxPool2d((3,3)),
            # nn.Conv2d(self.model.image_encoder.out_chans // 8, 3,(2,2), padding=(1,1)),
            # nn.MaxPool2d((2,2)),
            # LayerNorm2d(3),
            # nn.Upsample(size=(256,256))
            )
        self.bn = LayerNorm2d(3)
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(self.model.image_encoder.out_chans, self.model.image_encoder.out_chans // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(self.model.image_encoder.out_chans // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(self.model.image_encoder.out_chans // 4, self.model.image_encoder.out_chans // 8, kernel_size=2, stride=2),
        #     nn.GELU(),
        # )
        self.align_corners=False
    
    def create_transforms(self, new_size):
        Transforms = []
        new_h, new_w = new_size
        Transforms.append(A.Resize(int(new_h), int(new_w), interpolation=cv2.INTER_NEAREST))
        Transforms.append(ToTensorV2(p=1.0))
        return A.Compose(Transforms, p=1.)
    #sam 默认解码器
    def decoderfeatures(self,image_embeddings,x):
        image_pe = self.model.prompt_encoder.get_dense_pe()
        pos_src=torch.repeat_interleave(image_pe, x.shape[0], dim=0)
        output_tokens = torch.cat([self.model.mask_decoder.iou_token.weight, self.model.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens=output_tokens.unsqueeze(0).expand(1, -1, -1)
        src = torch.repeat_interleave(image_embeddings, output_tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.model.mask_decoder.transformer(src, pos_src, output_tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.model.mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.model.mask_decoder.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.model.mask_decoder.num_mask_tokens):
            hyper_in_list.append(self.model.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        features=torch.repeat_interleave(masks, 3, dim=1)
        return features
    #sam 训练新的解码器
    def decoderfeatures2(self,image_embeddings):
        return self.decoder(image_embeddings)
    def forward(self, x):
        with torch.no_grad():
            # img_1024 = (img_1024 - img_1024.min()) / np.clip(
            #     img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            # ) 
            # x=(x-torch.min(x))/torch.clip(torch.max(x)-torch.min(x),min=1e-8,max=None)
            # transformed_image = self.transform.apply_image_torch(x)
            # input_image = self.model.preprocess(transformed_image)
             # Transform the image to the form expected by the model
            x= self.model.preprocess(x)
            x = x.permute(0,2,3,1)
            images=[]
            for b in range(x.shape[0]):
                image=x[b].cpu().numpy()
                image =self.transform(image=image)
                image = torch.as_tensor(image["image"],device=self.model.device)
                images.append(image)
            images=torch.stack(images, dim=0)
            # images= self.model.preprocess(images)
            image_embeddings = self.model.image_encoder(images)
        features=self.decoderfeatures2(image_embeddings)
        
        return features
class sam_med2d_b(sam_med2d_base):
    def __init__(self,num_classes,user_to_head:bool=True):
         super(sam_med2d_b, self).__init__("vit_b",num_classes,user_to_head)
# class sam_l(sam_base):
#     def __init__(self,num_classes,user_to_head:bool=True):
#          super(sam_l, self).__init__("vit_l",num_classes,user_to_head)
# class sam_h(sam_base):
#     def __init__(self,num_classes,user_to_head:bool=True):
#          super(sam_h, self).__init__("vit_h",num_classes,user_to_head)