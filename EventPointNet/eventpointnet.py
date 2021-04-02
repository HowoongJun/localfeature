from lcore.hal import *
import EventPointNet.train as train
import EventPointNet.nets as nets
import numpy as np
import torch
from common.Log import DebugPrint

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("CEventPointNet Constructor!")

    def __del__(self):
        print("CEventPointNet Destructor!")

    def Open(self, bGPUFlag):
        self.__gpuCheck = bGPUFlag
        self.__device = "cuda" if self.__gpuCheck else "cpu"
        self.__oQueryModel = nets.CEventPointNet().to(self.__device)
        self.__oQueryModel.load_state_dict(torch.load("./checkpoints/checkpoint.pth"))
        self.__oQueryModel.eval()
        DebugPrint().info("Load Model Completed!")

    def Close(self):
        print("CEventPointNet Close!")

    def Write(self, db, dbPath):
        oTrain = train.CTrain()
        oTrain.Open(db=db, dbPath=dbPath)
        oTrain.Setting()
        oTrain.run()

    def Read(self):
        kp, desc = self.__oQueryModel.forward(self.__Image)
        return kp, desc

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = np.expand_dims(np.asarray(Value), axis=1)
            self.__Image = torch.from_numpy(self.__Image).to(self.__device, dtype=torch.float)
        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)

    def Reset(self):
        print("CEventPointNet Reset!")