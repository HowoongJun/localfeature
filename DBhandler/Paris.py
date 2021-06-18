from torch.utils.data import Dataset
import torch 
import os
from glob import glob
from skimage import io
from skimage.transform import resize
import numpy as np

class CDataset(Dataset):
    def __init__(self, dataPath, transforms=None):
        self.__transforms = transforms
        self.__dataPath = dataPath
        jpgDataList = [os.path.basename(x) for x in glob(dataPath + "*.jpg")]
        self.__dataList = jpgDataList

    def __len__(self):
        return len(self.__dataList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if(idx >= len(self.__dataList) or idx < 0):
            print("Maximum")
            idx = len(self.__dataList) - 1

        image = io.imread(self.__dataPath + self.__dataList[idx], as_gray=True)
        image = (resize(image, (480,640)) * 255).astype(np.uint8)
        image = np.expand_dims(image, axis=0)
        
        target_05 = (((image / 255.0) ** (1.0 / 0.5)) * 255).astype(np.uint8)
        target_04 = (((image / 255.0) ** (1.0 / 0.4)) * 255).astype(np.uint8)
        target_03 = (((image / 255.0) ** (1.0 / 0.3)) * 255).astype(np.uint8)
        target_02 = (((image / 255.0) ** (1.0 / 0.2)) * 255).astype(np.uint8)


        result = {'target': image, 'image0.5': target_05, 'image0.4': target_04, 'image0.3': target_03, 'image0.2': target_02}

        if self.__transforms:
            result = self.__transforms(result)
        
        return result