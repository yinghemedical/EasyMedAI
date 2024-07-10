import EasyMedAI
from EasyMedAI.dataset.MedMNIST_PLUS import MNIST_224,MNIST_128, MNISTSubSetType,MNIST_64
from EasyMedAI.dataset.segmentation.demoDataset import CamVid,create_palette
from EasyMedAI.dataset.MSD import MSDDataSet, MSDSubSetType
from EasyMedAI.enums import DataSetType, TaskType
from EasyMedAI.models.modelFactory import createTrainModel
from torch.hub import load
import medmnist 
from medmnist import OrganAMNIST
import torch
from EasyMedAI.utils import create_color_to_class


device = torch.device("cuda:0")
torch.cuda.set_device(device)

color_to_class=create_color_to_class([0,1])
trainModel =createTrainModel("sam_vit_l_pretrained",2,"resnet50")
train_set= MSDDataSet("data/msd",subSetName=MSDSubSetType.Task09_Spleen,task_type=TaskType.Segmentation,dataset_type=DataSetType.train,transform=trainModel.transform_img,target_transform=trainModel.transform_lable)
valid_set= MSDDataSet("data/msd",subSetName=MSDSubSetType.Task09_Spleen,task_type=TaskType.Segmentation,dataset_type=DataSetType.val,transform=trainModel.transform_img,target_transform=trainModel.transform_lable)

# train_dataset = OrganAMNIST(split="train",download=True,root="data/medmnist")

# import sys
# sys.path.append('../')

# from MedLvmWorkflow.dataset.segmentation.demoDataset import create_palette
# from MedLvmWorkflow.models.modelFactory import createTrainModel,createInferModel
# color_to_class=create_palette("data/CamVid/class_dict.csv")
# num_class=len(color_to_class.items())
# inferModel =createInferModel("deeplabv3_resnet50_pretrained",num_class,backboneModlePertrained="./notebooks/epoch_20.pth")


# color_to_class=create_palette("data/CamVid/class_dict.csv")
# num_class=len(color_to_class.items())

#trainModel =createTrainModel("deeplabv3_resnet50_pretrained",num_class)
# train_set = CamVid(
#     'data/CamVid',
#     img_folder='train',
#     mask_folder='train_labels',
#     transform=trainModel.transform_img,
#     target_transform=trainModel.transform_lable)

# valid_set = CamVid(
#     'data/CamVid',
#     img_folder='val',
#     mask_folder='val_labels',
#     transform=trainModel.transform_img,
#     target_transform=trainModel.transform_lable)



from EasyMedAI.trainClass.train import TrainTool
trainModel.optim["lr"]=0.0005
trainProcess=TrainTool()
trainProcess.startTrain(trainModel,
                        train_set,
                        valid_set,
                        batch_size=2,
                        color_to_class=color_to_class,max_epochs=100)