from lcore.hal import *
import numpy as np
from common.Log import DebugPrint
import cv2

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("ORB Constructor")
        self.__threshold = 3000
    
    def __del__(self):
        print("Close")

    def Open(self, gpu_flag, args_mode):
        DebugPrint().info("Open")

    def Close(self):
        print("Close")

    def Read(self):
        self.__oOrb = cv2.ORB_create(nfeatures=self.__threshold)
        kpt, desc = self.__oOrb.detectAndCompute(self.__oImage, None)
        return kpt, desc, None

    def Write(self):
        DebugPrint().info("Write")
    
    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__oImage = np.asarray(Value)
            self.__oImage = np.squeeze(self.__oImage, axis=0)

        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)

        elif(SetCmd == eSettingCmd.eSettingCmd_THRESHOLD):
            self.__threshold = np.uint16(Value)

    def Reset(self):
        DebugPrint().info("Reset")