from common.Log import DebugPrint
from lcore.hal import *
import numpy as np
import localfeature_ref.r2d2.extract as extract
from PIL import Image
import cv2

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("Constructor")

    def __del__(self):
        print("Destructor")

    def Open(self, gpu_flag, args_mode):
        self.__gpuCheck = gpu_flag
        self.__device = "cuda" if self.__gpuCheck else "cpu"

    def Close(self):
        print("Close")

    def Write(self):
        print("Write")

    def Read(self):
        kpt, vDesc, vScores = extract.extract_keypoints(self.__gpuCheck, "./localfeature_ref/r2d2/checkpoints/r2d2_WAF_N16.pt", self.__Image, threshold=self.__threshold)
        vKpt = []
  
        for kptNo in range(len(kpt)):
            vKpt_tmp = cv2.KeyPoint(int(kpt[kptNo][0]), int(kpt[kptNo][1]), 5.0)
            vKpt.append(vKpt_tmp)
        return vKpt, vDesc, vScores

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = np.asarray(Value, dtype=np.uint8)
            self.__Image = np.squeeze(self.__Image, axis=0)
            self.__Image = Image.fromarray(self.__Image, "RGB")
            
        elif(SetCmd == eSettingCmd.eSettingCmd_THRESHOLD):
            self.__threshold = np.uint8(Value)

    def Reset(self):
        print("Reset")