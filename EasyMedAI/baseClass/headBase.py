from mmengine.model import BaseModel
import torch.nn as nn
class HeadBase(BaseModel):
    def __init__(self, num_classes:int, backboneModel:nn.Module,loss:any,embedding_size:int,patch_size:int):
        super(HeadBase, self).__init__()
        self.loss=loss
        self.num_classes=num_classes
        self.backboneModel=backboneModel
        self.embedding_size=embedding_size
        self.patch_size = patch_size
    def forward(self, imgs, labels, mode):
        x = self.forward_torch(imgs)
        if mode == 'loss':
            return {'loss': self.loss}
        elif mode == 'predict':
            return x, labels
    def forward_torch(self, x):
        raise NotImplementedError()