from enum import Enum
class TaskType(Enum):
    Segmentation =1
    Classification =2
class DataSetLoadType(Enum):
    Png=1
    Voxel =2
    Voxel_3D =3

