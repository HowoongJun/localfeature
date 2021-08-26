import torch, cv2
from common.Log import DebugPrint
from lcore.hal import *
from localfeature_ref.superpoint.superpointfrontend import SuperPointFrontend
import numpy as np

class CModel(CVisualLocalizationCore):
    def __init__(self, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True):
        self.__uNmsDist = nms_dist
        self.__fConfThresh = conf_thresh
        self.__fNnThresh = nn_thresh

    def __del__(self):
        print("Destructor")

    def Open(self, gpu_flag, args_mode):
        self.__device = "cuda" if gpu_flag else "cpu"
        self.__oSPPFrontend = SuperPointFrontend(weights_path="./localfeature_ref/superpoint/checkpoints/checkpoint.pth",
                                nms_dist=self.__uNmsDist,
                                conf_thresh=self.__fConfThresh,
                                nn_thresh=self.__fNnThresh,
                                cuda=gpu_flag)
    
    def Close(self):
        print("Close")

    def Write(self):
        print("Write")

    def Read(self):
        with torch.no_grad():
            
            kpt, desc, heatmap = self.__oSPPFrontend.run(self.__Image)
            if(heatmap is None and desc is None):
                return kpt, None, None
            vKpt = []
            for kptNo in range(len(kpt[0])):
                vKpt_tmp = cv2.KeyPoint(int(kpt[0][kptNo]), int(kpt[1][kptNo]), 5.0)
                vKpt.append(vKpt_tmp)
            desc = np.transpose(desc)
            oHeatmap = ((heatmap - np.min(heatmap)) * 255 / (np.max(heatmap) - np.min(heatmap))).astype(np.uint8)
            return vKpt, desc, oHeatmap

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__Image = np.asarray(Value, dtype=np.float32)
            self.__Image = np.squeeze(self.__Image, axis=0)
            self.__Image = self.__Image.astype('float32') / 255.
            
    def Reset(self):
        self.__Image = None       
        