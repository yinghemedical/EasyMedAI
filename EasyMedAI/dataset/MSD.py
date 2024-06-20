from copy import deepcopy
import json
import os
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
from EasyMedAI import utils
from EasyMedAI.enums import DataSetLoadType, DataSetType, TaskType
from EasyMedAI.dataset.datasetBase import datasetBase
from enum import Enum
from PIL import Image
import nibabel
import tarfile
DATASETHOMEPAGE="http://medicaldecathlon.com"
class MSDSubSetType(Enum):
    Task01_BrainTumours= "Task01_BrainTumours"
    Task02_Heart= "Task02_Heart"
    Task03_Liver ="Task03_Liver"
    Task04_Hippocampus ="Task04_Hippocampus"
    Task05_Prostate ="Task05_Prostate"
    Task06_Lung ="Task06_Lung"
    Task07_Pancreas ="Task07_Pancreas"
    Task08_HepaticVessels ="Task08_HepaticVessels"
    Task09_Spleen ="Task09_Spleen"
    Task10_Colon ="Task10_Colon"
##记录lable在什么窗宽窗位下
window_config={"Task02_Heart":[{"window":[1873,936],"lables":[1]}],"Task09_Spleen":[{"window":[350,40],"lables":[1]}]}
def split_array(arr, ratios):
    # 计算拆分点
    split_points = [int(len(arr) * ratio) for ratio in np.cumsum(ratios)]
    # 拆分数组
    return [arr[start:end] for start, end in zip([0] + split_points, split_points + [len(arr)])]
 


class MSDDataSet(datasetBase):
    def __init__(self,
                 root,
                 subSetName:MSDSubSetType=None,
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
       
        self.original_name=self.subSetName+".tar"
        if not os.path.exists(os.path.join(self.root, self.subSetName)):
            os.makedirs(os.path.join(self.root, self.subSetName))
        #检查原始文件是否存在
        if not os.path.exists(os.path.join(self.root, self.subSetName,self.original_name)):
            self.download()
        if load_type==  DataSetLoadType.Png:    
            self.img_folder = os.path.join(self.root, self.subSetName,self.subSetName,"png")
            self.mask_folder = os.path.join(self.root, self.subSetName,self.subSetName,"mask")
            self.datasetInfofile=os.path.join(self.root, self.subSetName,self.subSetName,"pngDataSet.json")
        elif load_type==  DataSetLoadType.Voxel:
            self.img_folder = os.path.join(self.root, self.subSetName,self.subSetName,"voxel")
            self.mask_folder = os.path.join(self.root, self.subSetName,self.subSetName,"mask_voxel") 
            self.datasetInfofile=os.path.join(self.root, self.subSetName,self.subSetName,"voxelDataSet.json") 
        elif load_type==  DataSetLoadType.Voxel_3D:
            self.img_folder = os.path.join(self.root, self.subSetName,self.subSetName)
            self.mask_folder = os.path.join(self.root, self.subSetName,self.subSetName)
            self.datasetInfofile=os.path.join(self.root, self.subSetName,self.subSetName,"dataset.json") 
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
            os.makedirs(self.mask_folder,exist_ok=True)
        #需要初始化数据集
        if len(os.listdir(self.img_folder)) <=0:
            #解压原始文件
            if load_type==  DataSetLoadType.Png:    
                with tarfile.open(os.path.join(self.root, self.subSetName,self.original_name), "r:") as tar:
                    tar.extractall(path=os.path.join(self.root, self.subSetName))
                originalFolder = os.path.join(self.root, self.subSetName,self.subSetName)
                with open(originalFolder+"/dataset.json","r") as  fileIo:
                    dataSetInfo = json.load(fileIo)
                if self.dataset_type==DataSetType.train or self.dataset_type==DataSetType.val:
                    dataList=dataSetInfo["training"]
                    ratios = [0.8, 0.2]  # 指定拆分比例
                    splitted_arr = split_array(dataList, ratios)
                    trainList=splitted_arr[0]
                    trainListInfo=self.convertPngDataSet(trainList,originalFolder)
                    varList= splitted_arr[1]
                    varListInfo=self.convertPngDataSet(varList,originalFolder)
                    dataSetInfo={DataSetType.train.value:trainListInfo,DataSetType.val.value:varListInfo}
                    with open(self.datasetInfofile,"w") as file:
                        json.dump(dataSetInfo,file)
                    pass
        
        with open(self.datasetInfofile,"r") as file:
           dataSetInfo = json.load(file)
        convertDs=dataSetInfo[self.dataset_type.value]
        # if  self.dataset_type.value   
            # print(npz_file)     
        self.images=[item["image"] for item in convertDs]
        self.masks=[item["label"] for item in convertDs]
        # self.images = list(
        #     sorted(os.listdir(self.img_folder),key=lambda x: int(x.replace('.png',''))))
        # self.masks = list(
        #     sorted(os.listdir(self.mask_folder),key=lambda x: int(x.replace('.npy',''))))
    def getLable(self,index):
        return np.load(os.path.join(self.mask_folder,self.masks[index]))
    def convertPngDataSet(self,dataList,originalFolder):
        i=0
        subDataSetInfo=[]
        windowInfos=window_config[self.subSetName]
        for item in dataList:
            niffFile=nibabel.load(os.path.join(originalFolder,item["image"].replace("./","")))
            lableFile= nibabel.load(os.path.join(originalFolder,item["label"].replace("./","")))
            niffFile=niffFile.get_fdata()
            lableFile=lableFile.get_fdata()
            for j in range(niffFile.shape[2]):
                sdata=niffFile[:,:,j]
                ldata=lableFile[:,:,j].astype(np.float32)
                unique_elements = np.unique(ldata).astype(np.float32)
                s=0
                for windowInfo in windowInfos:
                    allLables=np.array(windowInfo["lables"],np.float32)
                    in_array=np.isin(allLables,unique_elements)
                    if in_array.max()==True: #表示在该层有lable信息，需要保存
                        fit=np.isin(ldata,allLables).astype(np.float32)
                        ldata=ldata*fit
                        ldata=np.rot90(ldata)
                        lableFileName=os.path.basename(item["image"])+"_i"+str(j)+"_w"+str(s)+".npy"
                        np.save(os.path.join(self.mask_folder,lableFileName),ldata.astype(np.uint8))
                        lableFileName=os.path.basename(item["image"])+"_i"+str(j)+"_w"+str(s)+".npy"
                        # image = Image.fromarray(ldata.astype('uint8')*255)
                        # image.save(os.path.join(self.img_folder,lableFileName+".png"))
                        pngData=utils.convertNiiToPng(sdata,windowInfo["window"][0],windowInfo["window"][1])
                        pngData=np.rot90(pngData)
                        image = Image.fromarray(pngData.astype('uint8'))
                        pngFileName=os.path.basename(item["image"])+"_i"+str(j)+"_w"+str(s)+".png"
                        image.save(os.path.join(self.img_folder,pngFileName))
                        subDataSetInfo.append({"image":pngFileName,"label":lableFileName})
                        pass
                    s=s+1
        return subDataSetInfo
    def download(self):
        raise RuntimeError(
            f"""
            Automatic download failed! Please download MSD manually.
            1. Go to {DATASETHOMEPAGE} and find the {self.original_name} Download link
            2. Download the file and put under your MSD root folder: 
                {os.path.join(self.root, self.subSetName)}
            """
        )