from lcore.hal import *
import EventPointNet.train as train
import EventPointNet.nets as nets
import numpy as np
import torch, cv2
from common.Log import DebugPrint

class CModel(CVisualLocalizationCore):
    def __init__(self):
        self.softmax = torch.nn.Softmax2d()
        self.__threshold = 3000

    def __del__(self):
        print("CEventPointNet Destructor!")

    def Open(self, bGPUFlag, argsmode):
        self.__gpuCheck = bGPUFlag
        self.__device = "cuda" if self.__gpuCheck else "cpu"
        self.__oQueryModel = nets.CEventPointNet().to(self.__device)
        if(argsmode == 'query' or argsmode == 'match'):
            if(self.__gpuCheck):
                self.__oQueryModel.load_state_dict(torch.load("./EventPointNet/checkpoints/checkpoint.pth"))
            else:
                self.__oQueryModel.load_state_dict(torch.load("./EventPointNet/checkpoints/checkpoint.pth", map_location=torch.device("cpu")))
            self.__oSift = cv2.SIFT_create()
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
            kptDist, _ = self.__oQueryModel.forward(self.__Image)
            kptDist = self.softmax(kptDist)
            kptDist = kptDist[:,:-1,:]
            kptDist = torch.nn.functional.pixel_shuffle(kptDist, 8)
            kptDist = kptDist.data.cpu().numpy()
            DebugPrint().info("Generate Local Feature, Threshold: " + str(self.__threshold))
            kpt, desc, heatmap = self.__GenerateLocalFeature(kptDist, self.__threshold)
            return kpt, desc, heatmap

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA):
            self.__ImageOriginal = np.asarray(Value)
            self.__Image = np.expand_dims(np.asarray(Value), axis=1)
            self.__Image = torch.from_numpy(self.__Image).to(self.__device, dtype=torch.float)
        elif(SetCmd == eSettingCmd.eSettingCmd_IMAGE_CHANNEL):
            self.__channel = np.uint8(Value)
        elif(SetCmd == eSettingCmd.eSettingCmd_THRESHOLD):
            self.__threshold = np.uint16(Value)

    def Reset(self):
        self.__Image = None
        self.__channel = None

    def __GenerateLocalFeature(self, keypoint_distribution, threshold):
        heatmap = np.squeeze(keypoint_distribution, axis=0)
        heatmap = np.squeeze(heatmap, axis=0)
        heatmap_aligned = heatmap.reshape(-1)
        heatmap_aligned = np.sort(heatmap_aligned)[::-1]
        xs, ys = np.where(heatmap >= heatmap_aligned[threshold])
        vKpt = []
        vDesc = []
        if(len(self.__ImageOriginal.shape) >= 3):
                self.__ImageOriginal = np.squeeze(self.__ImageOriginal, axis=0)
        for kptNo in range(len(xs)):
            vKpt_tmp = cv2.KeyPoint(int(ys[kptNo]), int(xs[kptNo]), 5.0)
            vKpt.append(vKpt_tmp)

        _, vDesc = self.__oSift.compute(self.__ImageOriginal, vKpt)
        
        return vKpt, vDesc, heatmap