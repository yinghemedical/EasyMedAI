from typing import Dict, Optional, Union
from torch.utils.data.dataset import Dataset
from mmengine.model import BaseModel
from EasyMedAI.dataset.datasetBase import datasetBase
from EasyMedAI.hocks.visualizerHocks import SegmentationVisHook
from EasyMedAI.models.modelFactory import MMModelInterface
from EasyMedAI.trainClass.baseTool.trainBase import TrainBase
from torch.optim import AdamW
from mmengine.optim import AmpOptimWrapper
from mmengine.logging  import MMLogger
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from datetime import datetime
class TrainTool:
    def startTrain(self,model:MMModelInterface,trainDataSet:datasetBase,valDataSet:datasetBase,testDataSet:Dataset=None,batch_size:int=32,max_epochs=5, val_interval=1,checkpoint_interval=10,work_dir='./train_work_dir',vis_backends=None,color_to_class={},experiment_name=None):
        optim_wrapper=dict(type=AmpOptimWrapper, optimizer=model.optim)
        train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)
        val_evaluator=[dict(type=item) for item in model.metrics]
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=checkpoint_interval,save_param_scheduler=False))
        #可视化钩子
        custom_hooks=[dict(type='SegmentationVisHook', color_to_class=color_to_class)]
       
        trainC=TrainBase()
        # now = datetime.now()
        # formatted_specific_date_time = now.strftime("%Y%m%d-%H:%M")
        if experiment_name == None:
            experiment_name=model.modelName+"_"+trainDataSet.name
        return trainC.train(model,
                                 trainDataSet,
                                 valDataSet,
                                 testDataSet,
                                 optim_wrapper=optim_wrapper,
                                 batch_size=batch_size,
                                 train_cfg=train_cfg ,
                                 val_evaluator=val_evaluator,
                                 custom_hooks=custom_hooks,
                                 default_hooks=default_hooks,
                                 work_dir=work_dir,
                                 
                                 experiment_name=experiment_name,
                                 vis_backends=vis_backends
                                 )