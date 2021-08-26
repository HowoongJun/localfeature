from common.Log import DebugPrint
from lcore.hal import *
import numpy as np
import cv2
from localfeature_ref.lfnet.run_lfnet import *

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("Constructor")

    def __del__(self):
        print("Destructor")

    def Open(self, gpu_flag, args_mode):
        print("Open")

    def Close(self):
        print("Close")

    def Write(self):
        print("Write")

    def Read(self):
        kpt, desc, _ = run(self.__Image)
        vKpt = []
        print(kpt.shape)
        for kptNo in range(len(kpt)):
            vKpt_tmp = cv2.KeyPoint(int(kpt[kptNo][0]), int(kpt[kptNo][1]), 5.0)
            vKpt.append(vKpt_tmp)
        
        return vKpt, desc, None

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__Image = np.asarray(Value, dtype=np.uint8)
        elif(SetCmd == eSettingCmd.eSettingCmd_THRESHOLD):
            self.__Threshold = Value

    def Reset(self):
        print("Reset lfnet")