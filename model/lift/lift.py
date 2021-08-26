from common.Log import DebugPrint
from lcore.hal import *
import numpy as np
import cv2
from localfeature_ref.lift.python.compute_detector import compute_detector

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("Constructor")

    def __del__(self):
        print("Destructor")

    def Open(self):
        print("Open")

    def Close(self):
        print("Close")

    def Write(self):
        print("Write")

    def Read(self):
        kpt = compute_detector(self.__Image)

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__Image = np.asarray(Value, dtype=np.uint8)

    def Reset(self):
        print("Reset")