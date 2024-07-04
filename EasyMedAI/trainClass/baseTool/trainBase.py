from typing import List, Optional, Union,Dict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from torch.optim import SGD
from mmengine.runner import Runner
from mmengine.model import BaseModel
from mmengine.visualization import Visualizer
from mmengine.hooks import Hook
from mmengine.visualization import AimVisBackend
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.evaluator import Evaluator
class TrainBase:
    #基础训练类
    def train(self,model:BaseModel,trainDataSet:Dataset,valDataSet:Dataset,testDataSet:Dataset=None, dataSetShuffle:bool=True, batch_size:int=32,work_dir:str="./train_work_dir",optim_wrapper:Optional[Union[OptimWrapper, Dict]] =dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),train_cfg:Optional[Dict]=dict(by_epoch=True, max_epochs=5, val_interval=1),val_cfg:Optional[Dict]=dict(),val_evaluator:Optional[Union[Evaluator, Dict, List]]=dict(),test_evaluator:Optional[Union[Evaluator, Dict, List]]=None,test_cfg:Optional[Dict]=None,custom_hooks: Optional[List[Union[Hook, Dict]]] = None,default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,experiment_name = None,vis_backends=[dict(type='AimVisBackend')]):
        train_dataloader = DataLoader(batch_size=batch_size,
                                    shuffle=dataSetShuffle,
                                    dataset=trainDataSet,drop_last=True)
        val_dataloader =  DataLoader(batch_size=batch_size,
                                    shuffle=False,
                                    dataset=valDataSet,drop_last=True)
        if testDataSet:
            test_dataloader= DataLoader(batch_size=batch_size,
                                        shuffle=dataSetShuffle,
                                        dataset=testDataSet)
        else:
            test_dataloader = None
         #训练可视化后端
        if vis_backends ==None:
            vis_backends=[dict(type='AimVisBackend',save_dir=work_dir+"/"+experiment_name)]
        visualizer=dict(type='Visualizer', vis_backends=vis_backends)
        runner = Runner(
            model=model,
            work_dir=work_dir+"/"+experiment_name,
            train_dataloader=train_dataloader,
            # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
            optim_wrapper=optim_wrapper,
            train_cfg=train_cfg,
            val_dataloader=val_dataloader,
            val_cfg=val_cfg,
            val_evaluator=val_evaluator,
            test_dataloader=test_dataloader,
            test_evaluator=test_evaluator,
            test_cfg=test_cfg,
            log_processor=dict(window_size=100, by_epoch=True, custom_cfg=None, num_digits=4),
            visualizer=visualizer,
            custom_hooks=custom_hooks,
            default_hooks=default_hooks,
            experiment_name=experiment_name,
        )
        # for visName in runner.visualizer._vis_backends:
        #     if visName=="AimVisBackend":
        #         vis: AimVisBackend=runner.visualizer._vis_backends[visName]
        #         vis.experiment.
        #         print(vis.experiment)
        #     pass
        # if vis_backends ==None:
        #     vis_backends=[dict(type='AimVisBackend',init_kwargs=dict(experiment=experiment_name))]
        # runner.visualizer.draw_binary_masks
        runner.logger.name="EasyMedAI"
        return runner.train()
