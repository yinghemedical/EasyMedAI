import os
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
from EasyMedAI.enums import DataSetLoadType, DataSetType, TaskType
from EasyMedAI.dataset.datasetBase import datasetBase
from medmnist.info import INFO
from enum import Enum
# DataSetConifg_2D={"pathmnist":{"downloadUrl":None,"size":224},"chestmnist":{"downloadUrl":"https://zenodo.org/records/10519652/files/chestmnist_224.npz?download=1","size":224}}
# DataSetConifg_2D_128={"pathmnist":{"downloadUrl":None,"size":128},"chestmnist":{"downloadUrl":None,"size":128}}
# DataSetConifg_2D_64={"pathmnist":{"downloadUrl":None,"size":64},"chestmnist":{"downloadUrl":None,"size":64}}
DATASETHOMEPAGE="https://github.com/MedMNIST/MedMNIST/"
window_config={"organamnist":[{1:[1200,-600]}]}
class MNISTSubSetType(Enum):
    pathmnist ="pathmnist"
    chestmnist= "chestmnist"
    dermamnist ="dermamnist"
    octmnist ="octmnist"
    pneumoniamnist = "pneumoniamnist"
    retinamnist ="retinamnist"
    breastmnist ="breastmnist"
    bloodmnist ="bloodmnist"
    tissuemnist ="tissuemnist"
    organamnist ="organamnist"
    organcmnist ="organcmnist"
    organsmnist ="organsmnist"
    # organmnist3d ="organmnist3d"
    # nodulemnist3d ="nodulemnist3d"
    # adrenalmnist3d ="adrenalmnist3d"
    # fracturemnist3d ="FractureMNIST3D"
    # vesselmnist3d ="vesselmnist3d"
    # synapsemnist3d ="synapsemnist3d"
# 0:
# 'train_images'
# 1:
# 'train_labels'
# 2:
# 'val_images'
# 3:
# 'val_labels'
# 4:
# 'test_images'
# 5:
# 'test_labels'
class MNIST(datasetBase):
    def __init__(self,
                 root,
                 subSetName:MNISTSubSetType=None,
                 size=224,
                 transform=None,
                 target_transform=None,
                 load_type:DataSetLoadType= DataSetLoadType.Png,
                 task_type:TaskType =TaskType.Segmentation,
                 dataset_type:DataSetType= DataSetType.train
                 ):
        super().__init__(
            root, transform=transform, target_transform=target_transform,load_type=load_type,task_type=task_type)
        self.subSetName=subSetName.value
        self.dataset_type=dataset_type
        self.info=INFO[self.subSetName]
        if task_type==TaskType.Segmentation :
            raise RuntimeError(
                f"""
               MNIST dataset  not support {TaskType.Segmentation.name}.
                """
            )
        if not load_type == DataSetLoadType.Png:
             raise RuntimeError(
                f"""
               MNIST dataset just support png.
                """
            )
        self.size_flag = f"_{size}"
        self.original_name=self.subSetName+self.size_flag+".npz"
        if not os.path.exists(os.path.join(self.root, self.subSetName)):
            os.makedirs(os.path.join(self.root, self.subSetName))
        #检查原始文件是否存在
        if not os.path.exists(os.path.join(self.root, self.subSetName,self.original_name)):
            self.download()
        if load_type==  DataSetLoadType.Png:    
            self.img_folder = os.path.join(self.root, self.subSetName,self.subSetName+self.size_flag,"png",self.dataset_type.value)
            self.mask_folder = os.path.join(self.root, self.subSetName,self.subSetName+self.size_flag,"class",self.dataset_type.value)
        elif load_type==  DataSetLoadType.Voxel:
            self.img_folder = os.path.join(self.root, self.subSetName,self.subSetName+self.size_flag,"voxel",self.dataset_type.value)
            self.mask_folder = os.path.join(self.root, self.subSetName,self.subSetName+self.size_flag,"mask_voxel",self.dataset_type.value)  
        elif load_type==  DataSetLoadType.Voxel_3D:
            self.img_folder = os.path.join(self.root, self.subSetName,self.subSetName+self.size_flag,"voxel_3d",self.dataset_type.value)
            self.mask_folder = os.path.join(self.root, self.subSetName,self.subSetName+self.size_flag,"mask_voxel_3d",self.dataset_type.value)
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
            os.makedirs(self.mask_folder)
        #加载原始文件
        if len(os.listdir(self.img_folder)) <=0:
            npz_file = np.load(
                os.path.join(os.path.join(self.root, self.subSetName,self.original_name)),
                mmap_mode="r",
            )
            imgs = npz_file[f"{dataset_type.value}_images"]
            lables =npz_file[f"{dataset_type.value}_labels"]
            for i in range(imgs.shape[0]):
                img=imgs[i]
                img = Image.fromarray(img)
                img = img.convert("RGB")
                img.save(os.path.join(self.img_folder,str(i)+".png"))
                np.save(os.path.join(self.mask_folder,str(i)+".npy"), lables[i])
            imgs=None
            lables= None
            
            # print(npz_file)     
        self.images = list(
            sorted(os.listdir(self.img_folder),key=lambda x: int(x.replace('.png',''))))
        self.masks = list(
            sorted(os.listdir(self.mask_folder),key=lambda x: int(x.replace('.npy',''))))
    
    def getLable(self,index):
        tasks=self.info["task"].split(",")
        mask_path = os.path.join(self.mask_folder,
                                 self.masks[index])
        classData= np.load(mask_path)
        if  "multi-label" in tasks :
            return  classData
        else:
            return  [classData]
    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(
                url=self.info[f"url{self.size_flag}"],
                root=self.root,
                filename=self.original_name,
                md5=self.info[f"MD5{self.size_flag}"],
            )
        except:
            raise RuntimeError(
                f"""
                Automatic download failed! Please download {self.subSetName}{self.size_flag}.npz manually.
                1. [Optional] Check your network connection: 
                    Go to {DATASETHOMEPAGE} and find the Zenodo repository
                2. Download the npz file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url{self.size_flag}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5{self.size_flag}"]}
                4. Put the npz file under your MedMNIST root folder: 
                    {os.path.join(self.root, self.subSetName)}
                """
            )

    def __len__(self):
        return len(self.images)


class MNIST_224(MNIST):
    def __init__(self,
                 root,
                 subSetName:MNISTSubSetType=MNISTSubSetType.organamnist,
                 transform=None,
                 target_transform=None,
                 load_type:DataSetLoadType= DataSetLoadType.Png,
                 task_type:TaskType =TaskType.Segmentation,
                 dataset_type:DataSetType= DataSetType.train
                 ):
        super().__init__(
            root,subSetName=subSetName,size=224, transform=transform, target_transform=target_transform,load_type=load_type,task_type=task_type,dataset_type=dataset_type)
class MNIST_128(MNIST):
    def __init__(self,
                 root,
                 subSetName:MNISTSubSetType=MNISTSubSetType.organamnist,
                 transform=None,
                 target_transform=None,
                 load_type:DataSetLoadType= DataSetLoadType.Png,
                 task_type:TaskType =TaskType.Segmentation,
                 dataset_type:DataSetType= DataSetType.train
                 ):
        super().__init__(
            root,subSetName=subSetName,size=128, transform=transform, target_transform=target_transform,load_type=load_type,task_type=task_type,dataset_type=dataset_type)
class MNIST_64(MNIST):
    def __init__(self,
                 root,
                 subSetName="",
                 transform=None,
                 target_transform=None,
                 load_type:DataSetLoadType= DataSetLoadType.Png,
                 task_type:TaskType =TaskType.Segmentation,
                 dataset_type:DataSetType= DataSetType.train
                 ):
        super().__init__(
            root,subSetName=subSetName,size=64, transform=transform, target_transform=target_transform,load_type=load_type,task_type=task_type,dataset_type=dataset_type)

