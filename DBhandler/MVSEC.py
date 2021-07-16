###
#
#       @Brief          MVSEC.py
#       @Details        Multi Vehicle Stereo Event Camera dataset handler
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 26, 2021
#       @Version        v0.13
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

        homSize = random.uniform(0.5, 1.0)
        homProj = random.uniform(-0.003, 0.003)
        hommatrixsize = np.array([[homSize, 0.0, 0.0], [0.0, homSize, 0.0], [0.0, 0.0, 1.0]])
        hommatrixproj = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [homProj, homProj, 1.0]])
        transformsize = transform.ProjectiveTransform(matrix=hommatrixsize)
        transformproj = transform.ProjectiveTransform(matrix=hommatrixproj)
        rotDeg = random.randrange(-90, 90)
        
        image = npyData['image']
        rotimage = transform.rotate(image, rotDeg)
        homimage = transform.warp(image, transformsize.inverse)
        homimage = transform.warp(homimage, transformproj.inverse)

        image = np.expand_dims(image, axis=0)
        rotimage = np.expand_dims(rotimage, axis=0)
        homimage = np.expand_dims(homimage, axis=0)

        target = npyData['tsimagenormalized']
        rottarget = transform.rotate(target, rotDeg)
        homtarget = transform.warp(target, transformsize.inverse)
        homtarget = transform.warp(homtarget, transformproj.inverse)

        target = np.expand_dims(target, axis=0)
        rottarget = np.expand_dims(rottarget, axis=0)
        homtarget = np.expand_dims(homtarget, axis=0)

        bright0 = (((image / 255.0) ** (1.0 / 1.5)) * 255).astype(np.uint8)
        bright1 = (((image / 255.0) ** (1.0 / 2.0)) * 255).astype(np.uint8)
        bright2 = (((image / 255.0) ** (1.0 / 2.5)) * 255).astype(np.uint8)

        dark0 = (((image / 255.0) ** (1.0 / 0.8)) ** 255).astype(np.uint8)
        dark1 = (((image / 255.0) ** (1.0 / 0.5)) ** 255).astype(np.uint8)
        dark2 = (((image / 255.0) ** (1.0 / 0.2)) ** 255).astype(np.uint8)

        result = {'image': image, 'target': target,
                  'rotimage': rotimage, 'rottarget': rottarget,
                  'homimage': homimage, 'homtarget': homtarget,
                  'bright0': bright0, 'bright1': bright1, 'bright2': bright2,
                  'dark0': dark0, 'dark1': dark1, 'dark2': dark2}

        if self.__transforms:
            result = self.__transforms(result)
        
        return result