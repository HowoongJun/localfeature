###
#
#       @Brief          kitti.py
#       @Details        DB handler for KITTI dataset
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Aug. 06, 2021
#       @Version        v0.1
#
###

import numpy as np
from glob import glob
import cv2, os
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ImageTopic():
    def __init__(self, Pose:np.array, Image:np.array):
        self.__pose = Pose
        self.__image = Image
    
    @property
    def Image(self) -> np.array:
        return self.__image
    
    @Image.setter
    def Image(self, Image:np.array) -> None:
        self.__image = Image

    @property
    def Pose(self) -> np.array:
        return self.__pose

    @Pose.setter
    def Pose(self, Pose:np.array) -> None:
        self.__pose = Pose

class CKitti():
    def __init__(self, dataset):
        self.__ImagePath = "/root/Workspace/dataset/KITTI/dataset/" + str(dataset) + "/image_2/"
        self.__PosePath = "/root/Workspace/dataset/KITTI/poses/" + str(dataset) + ".txt"
        self.__CalibPath = "/root/Workspace/dataset/KITTI/dataset/" + str(dataset) + "/calib.txt"
        self.__TotalImageNumber = 0
        self.getImageList()

    def getPose(self):
        poses = []
        with open(self.__PosePath, 'r') as f:
            lines = f.readlines()
            if(self.__TotalImageNumber != 0):
                lines = [lines[i] for i in range(self.__TotalImageNumber)]

            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                poses.append(T_w_cam0)
        return poses

    def getCalib(self):
        data = {}
        with open(self.__CalibPath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                    # P_rect_00 = np.reshape(data['P0'], (3, 4))
                    # P_rect_10 = np.reshape(data['P1'], (3, 4))
                    # P_rect_20 = np.reshape(data['P2'], (3, 4))
                    # P_rect_30 = np.reshape(data['P3'], (3, 4))
                except ValueError:
                    pass
        return data

    def getImageList(self):
        self.__ImageList = [os.path.basename(x) for x in glob(self.__ImagePath + "*.png")]
        self.__TotalImageNumber = len(self.__ImageList)
        return self.__ImageList

    def getImage(self, strImageName):
        return cv2.imread(self.__ImagePath + strImageName)

    def getAllData(self):
        PoseData = self.getPose()
        ImageData = []
        for i in range(len(self.__ImageList)):
            ImageData.append(ImageTopic(PoseData[i], self.getImage(self.__ImageList[i])))
        return ImageData

    def getGT(self):
        PoseData = self.getPose()
        initPoint = [0, 0, 0, 1]
        X, Z = [], []
        for i in range(1, len(PoseData)):
            Point = PoseData[i].dot(initPoint)
            X.append(Point[0])
            Z.append(Point[2])
        return [X, Z]

  
