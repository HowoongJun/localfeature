###
#
#       @Brief          MVSEC.py
#       @Details        Multi Vehicle Stereo Event Camera dataset handler
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 26, 2021
#       @Version        v0.12
#
###


import torch
from torch.utils.data import Dataset
from glob import glob
import os
from skimage import io
from skimage import transform
import numpy as np
import random

class CDataset(Dataset):
    def __init__(self, dataPath, transforms=None):        
        self.__transforms = transforms
        self.__targetPath = dataPath + "/tsevent/"
        self.__trainPath = dataPath + "/image/"
        self.__dataPath = dataPath + "/dataloader/"
        self.__trainImageList = [os.path.basename(x) for x in glob(self.__trainPath + "*.png")]
        self.__targetList = [os.path.basename(x) for x in glob(self.__targetPath + "*.png")]
        self.__dataList = [os.path.basename(x) for x in glob(self.__dataPath + "*.npz")]
        self.__trainImageList.sort()
        self.__targetList.sort()

    def __len__(self):
        return len(self.__targetList)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if(idx >= len(self.__dataList) or idx < 0):
            print("Maximum")
            idx = len(self.__dataList) - 1
        npyData = np.load(self.__dataPath + self.__dataList[idx])

        rotDeg = random.randrange(-90, 90)
        # image = io.imread(self.__trainPath + str(npyData['image']))
        image = npyData['image']
        rotimage = transform.rotate(image, rotDeg)
        
        image = np.expand_dims(image, axis=0)
        rotimage = np.expand_dims(rotimage, axis=0)
        
        target = npyData['tsimagenormalized']
        rottarget = transform.rotate(target, rotDeg)

        target = np.expand_dims(target, axis=0)
        rottarget = np.expand_dims(rottarget, axis=0)
        
        result = {'image': image, 'target': target, 'rotimage': rotimage, 'rottarget': rottarget}

        if self.__transforms:
            result = self.__transforms(result)
        
        return result