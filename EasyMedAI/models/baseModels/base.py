from typing import Dict, Optional, Union
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from torch.optim import AdamW
from mmengine.optim import AmpOptimWrapper
class LvmBaseModel(torch.nn.Module):
    def __init__(self,num_classes:int,user_to_head:bool=False):
        super(LvmBaseModel, self).__init__()
        #子类实现
        self.transform_img=None
        self.transform_lable=None
        self.num_classes = num_classes
        self.metrics:list[BaseMetric]=None
        self.optim=dict(type=AdamW, lr=2e-4)
    def lossFun(self,x,data_samples):
        raise NotImplementedError