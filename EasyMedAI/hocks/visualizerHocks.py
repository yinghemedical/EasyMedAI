from mmengine.hooks import Hook
import shutil
import cv2
import os.path as osp
import mmengine.visualization
import torch
import numpy as np
import os
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import mmengine
from PIL import Image
from EasyMedAI.enums import DataSetLoadType
from mmengine.visualization import Visualizer
import torch.nn.functional as F
@HOOKS.register_module()
class SegmentationVisHook(Hook):

    def __init__(self, vis_num=1,color_to_class={}) -> None:
        super().__init__()
        self.vis_num = vis_num
        self.color_to_class = color_to_class

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch=None,
                       outputs=None) -> None:
        # if batch_idx > self.vis_num:
        #     return
        
        preds, data_samples,featmaps = outputs
        img_paths = data_samples['img_path']
        mask_paths = data_samples['mask_path']
        _, C, H, W = preds.shape
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path,load_type,featmap) in enumerate(zip(preds, img_paths, mask_paths,data_samples["load_type"],featmaps)):
            # pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            
            if load_type==DataSetLoadType.Png.name:
                image = cv2.imread(img_path)
                gt_mask= np.load(mask_path)
                # gt_mask=Image.fromarray(gt_mask)
                gt_image=cv2.cvtColor(gt_mask,cv2.COLOR_GRAY2BGR)
                gt_image=cv2.resize(gt_image,(image.shape[1],image.shape[0]))
                for color, class_id in self.color_to_class.items():
                    tp=gt_mask == class_id
                    gt_image[:,:,0][tp]=color[2]
                    gt_image[:,:,1][tp]=color[1]
                    gt_image[:,:,2][tp]=color[0]
                
            # image=cv2.resize(image,(W,H))
            # image = image.resize((H,W))
            runner.visualizer.set_image(image)
            for color, class_id in self.color_to_class.items():
                tp=pred == class_id
                # nparray = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                tp=np.asarray(tp.cpu().numpy(),dtype=np.uint8)
                # tp=cv2.form.cvtColor(,cv2.COLOR_RGB2GRAY)
                tp=cv2.resize(tp,(image.shape[1],image.shape[0]))
                runner.visualizer.draw_binary_masks(
                    tp==1,
                    colors=[color],
                    alphas=0.6,
                )
            # Convert RGB to BGR
            pred_mask = runner.visualizer.get_image()[..., ::-1]
            gt_mask = cv2.addWeighted(image,1,gt_image,0.6,0)
            # featmap=F.interpolate(featmap,size=(image.shape[1],image.shape[0]), mode="bilinear", align_corners=True)
            drawn_img =runner.visualizer.draw_featmap(featmap,overlaid_image=image,resize_shape=(image.shape[1],image.shape[0]), channel_reduction='select_max',alpha=0.8)
            c=np.vstack((image,gt_mask,drawn_img,pred_mask))
            for name,backend in runner.visualizer._vis_backends.items():
                # backend.add_image(f'pred_{osp.basename(img_path)}',image=pred_mask,step=runner.epoch)
                # backend.add_image(f'{osp.basename(img_path)}_featmp',image=drawn_img,step=runner.epoch)
                backend.add_image(f'{osp.basename(img_path)}',image=c,step=runner.epoch)
                # backend.add_image(f'gt_{osp.basename(img_path)}',image=gt_mask,step=runner.epoch)
            #drawn_img =runner.visualizer.draw_featmap(featmap, channel_reduction='select_max')
            # saved_dir = osp.join(runner.log_dir, 'val_vis_data', str(runner.epoch))
            # os.makedirs(saved_dir, exist_ok=True)

            # # shutil.copyfile(img_path,
            # #                 osp.join(saved_dir, osp.basename(img_path)))
            # # shutil.copyfile(mask_path,
            # #                 osp.join(saved_dir, osp.basename(mask_path)))
            # cv2.imwrite(
            #     osp.join(saved_dir, f'pred_{osp.basename(img_path)}'),
            #     pred_mask)