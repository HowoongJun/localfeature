from common.Log import DebugPrint
from lcore.hal import *
import numpy as np
import localfeature_ref.keynet.extract_multiscale_features as extract
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
        kpt, vDesc = extract.extract_keypoints(self.__Image)
        vKpt = []
        for kptNo in range(len(kpt)):
            vKpt_tmp = cv2.KeyPoint(int(kpt[kptNo][0]), int(kpt[kptNo][1]), 5.0)
            vKpt.append(vKpt_tmp)
        oHeatmap = []
        return vKpt, vDesc, oHeatmap

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)
        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__Image = np.asarray(Value, dtype=np.uint8)
            
            
    def Reset(self):
        print("Reset")