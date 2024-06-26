from mmengine.model import BaseModel

from EasyMedAI.models.lvmModels.dinov2 import dinov2_s_pretrained
from EasyMedAI.models.segmentation.conv import conv_s
from EasyMedAI.models.segmentation.deeplabv3_resnet50 import Resnet50_pretrained
import torch
modesConfig={"deeplabv3_resnet50_pretrained":{"class":Resnet50_pretrained,"useBockbone":True,"useHead":True},
             "dinov2_s_pretrained":{"class":dinov2_s_pretrained,"useBockbone":True,"useHead":False},
            "conv_s":{"class":conv_s,"useBockbone":False,"useHead":True},
             
             }
from EasyMedAI.models.baseModels.base import LvmBaseModel
def createTrainModel(backboneModle:str,num_classes:int,headModel:str=None,backboneModlePertrained:str=None,headModlePertrained:str=None):
    """
    create Train Model by  Name
    """

    if headModel:
        backboneModleConfig=modesConfig[backboneModle]
        headModelConfig=modesConfig[headModel]
        return createCusTrainModel(backboneModleConfig["class"],num_classes,headModel=headModelConfig["class"])
    else:
        backboneModleConfig=modesConfig[backboneModle]
        return createCusTrainModel(backboneModleConfig["class"],num_classes)

def createCusTrainModel(backboneModle:type[LvmBaseModel],num_classes:int,headModel:type[LvmBaseModel]=None):
    """
    create Train Model by  class
    """
    if headModel:
        mModel= backboneModle(num_classes,user_to_head=False)
        mModel=mModel.eval()
        sModel= headModel(num_classes,user_to_head=True)
        return MMModelInterface(ModelInterface(mModel,sModel))
    else:
        mModel= backboneModle(num_classes,user_to_head=False)
        return MMModelInterface(ModelInterface(mModel))
def createInferModel(backboneModle:str,num_classes:int,headModel:str=None,backboneModlePertrained:str=None,headModlePertrained:str=None):
    """
    create Infer Model by  Name
    """
    if headModel:
        backboneModleConfig=modesConfig[backboneModle]
        headModelConfig=modesConfig[headModel]
        return createCusInferModel(backboneModleConfig["class"],num_classes,headModel=headModelConfig["class"],backboneModlePertrained=backboneModlePertrained,headModlePertrained=headModlePertrained)
    else:
        backboneModleConfig=modesConfig[backboneModle]
        return createCusInferModel(backboneModleConfig["class"],num_classes,backboneModlePertrained=backboneModlePertrained)
def createCusInferModel(backboneModle:type[LvmBaseModel],num_classes:int,headModel:type[LvmBaseModel]=None,backboneModlePertrained:str=None,headModlePertrained:str=None):
    """
    create Infer Model by  class
    """
    if headModel:
        mModel= backboneModle(num_classes,user_to_head=False)
        state_dict=torch.load(backboneModlePertrained)
        mModel.load_state_dict(state_dict)
        sModel= headModel(num_classes,user_to_head=True)
        state_dict=torch.load(headModlePertrained)
        sModel.load_state_dict(state_dict)
        model =ModelInterface(mModel,sModel).eval()
        return model
    else:
        mModel= backboneModle(num_classes,user_to_head=False)
        state_dict=torch.load(backboneModlePertrained)
        model =ModelInterface(mModel)
        model.load_state_dict(state_dict["state_dict"]["model"])
        model.eval()
        return model
class MMModelInterface(BaseModel):
    def __init__(self, model:LvmBaseModel):
        super(MMModelInterface, self).__init__()
        self.loss=model.lossFun
        self.num_classes = model.num_classes
        self.model=model
        # self.model.load_state_dict()
        self.transform_img=self.model.transform_img
        self.transform_lable=self.model.transform_lable
        self.metrics=self.model.metrics
        self.optim=self.model.optim
    def forward(self, imgs, data_samples=None, mode='tensor'):
        x,ft = self.model(imgs)
        if mode == 'loss':
            return {'loss': self.loss(x, data_samples)}
        elif mode == 'predict':
            return x, data_samples,ft
class ModelInterface(LvmBaseModel):
    def __init__(self, backboneModle:LvmBaseModel,headModel:LvmBaseModel=None):
        super(ModelInterface, self).__init__(0,user_to_head=False)
        if headModel:
            self.lossFun=headModel.lossFun
            self.num_classes = headModel.num_classes
        else:
            self.lossFun=backboneModle.lossFun
            self.num_classes = backboneModle.num_classes
        self.backboneModle=backboneModle
        self.headModel=headModel
        if headModel:
            self.transform_img=self.backboneModle.transform_img
            self.transform_lable=self.backboneModle.transform_lable
            self.metrics=self.headModel.metrics
            self.optim=self.headModel.optim
        else:
            self.transform_img=self.backboneModle.transform_img
            self.transform_lable=self.backboneModle.transform_lable
            self.metrics=self.backboneModle.metrics
            self.optim=self.backboneModle.optim
    def forward(self, imgs):
        if self.headModel:
            ft= self.backboneModle(imgs)
            x= self.headModel(ft)
            return x,ft
        else:
            x= self.backboneModle(imgs)
            return x,imgs