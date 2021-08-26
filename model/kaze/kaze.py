from lcore.hal import *
import numpy as np
from common.Log import DebugPrint
import cv2

class CModel(CVisualLocalizationCore):
    def __init__(self):
        self.__oKaze = cv2.KAZE_create()

    def __del__(self):
        print("Close")

    def Open(self, gpu_flag, args_mode):
        DebugPrint().info("Open KAZE")

    def Close(self):
        print("Close")

    def Read(self):
        kpt, desc = self.__oKaze.detectAndCompute(self.__oImage, None)
        return kpt, desc, None

    def Write(self):
        print("Write")

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__oImage = np.asarray(Value)
            self.__oImage = np.squeeze(self.__oImage, axis=0)

    def Reset(self):
        print("Reset")