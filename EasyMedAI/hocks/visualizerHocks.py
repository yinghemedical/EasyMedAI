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
        if batch_idx > self.vis_num:
            return
        preds, data_samples = outputs
        img_paths = data_samples['img_path']
        mask_paths = data_samples['mask_path']
        _, C, H, W = preds.shape
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path) in enumerate(zip(preds, img_paths, mask_paths)):
            pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            image = cv2.imread(img_path)
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
            gt_mask = cv2.imread(mask_path)
            gt_mask=cv2.resize(gt_mask,(image.shape[1],image.shape[0]))
            gt_mask = cv2.addWeighted(image,1,gt_mask,0.6,0)
            c=np.vstack((image,gt_mask,pred_mask))
            for name,backend in runner.visualizer._vis_backends.items():
                # backend.add_image(f'pred_{osp.basename(img_path)}',image=pred_mask,step=runner.epoch)
                backend.add_image(f'{osp.basename(img_path)}',image=c,step=runner.epoch)
                # backend.add_image(f'gt_{osp.basename(img_path)}',image=gt_mask,step=runner.epoch)
            # saved_dir = osp.join(runner.log_dir, 'val_vis_data', str(runner.epoch))
            # os.makedirs(saved_dir, exist_ok=True)

            # # shutil.copyfile(img_path,
            # #                 osp.join(saved_dir, osp.basename(img_path)))
            # # shutil.copyfile(mask_path,
            # #                 osp.join(saved_dir, osp.basename(mask_path)))
            # cv2.imwrite(
            #     osp.join(saved_dir, f'pred_{osp.basename(img_path)}'),
            #     pred_mask)