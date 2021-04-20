from lcore.hal import *
import EventPointNet.train as train
import EventPointNet.nets as nets
import numpy as np
import torch
from common.Log import DebugPrint

class CModel(CVisualLocalizationCore):
    def __init__(self):
        self.softmax = torch.nn.Softmax2d()

    def __del__(self):
        print("CEventPointNet Destructor!")

    def Open(self, bGPUFlag, argsmode):
        self.__gpuCheck = bGPUFlag
        self.__device = "cuda" if self.__gpuCheck else "cpu"
        self.__oQueryModel = nets.CEventPointNet().to(self.__device)
        if(argsmode == 'query'):
            self.__oQueryModel.load_state_dict(torch.load("./EventPointNet/checkpoints/checkpoint.pth"))
            DebugPrint().info("Load Model Completed!")

    def Close(self):
        print("CEventPointNet Close!")

    def Write(self, db, dbPath):
        oTrain = train.CTrain()
        oTrain.Open(db=db, dbPath=dbPath)
        oTrain.Setting()
        oTrain.run()

    def Read(self):
        with torch.no_grad():
            self.__oQueryModel.eval()
            kpt, desc = self.__oQueryModel.forward(self.__Image)
            kpt = self.softmax(kpt)
            kpt = kpt[:,:-1,:]
            kpt = torch.nn.functional.pixel_shuffle(kpt, 8)
            kpt = kpt.data.cpu().numpy()
            return kpt, desc

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__Image = np.expand_dims(np.asarray(Value), axis=1)
            self.__Image = torch.from_numpy(self.__Image).to(self.__device, dtype=torch.float)
        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)

    def Reset(self):
        self.__Image = None
        self.__channel = None