from lcore.hal import *
import EventPointNet.train as train

class CModel(CVisualLocalizationCore):
    def __init__(self):
        print("CEventPointNet Constructor!")

    def __del__(self):
        print("CEventPointNet Destructor!")

    def Open(self, bGPUFlag):
        self.__gpuCheck = bGPUFlag

    def Close(self):
        print("CEventPointNet Close!")

    def Write(self):
        oTrain = train.CTrain()
        oTrain.Setting()
        oTrain.run()

    def Read(self):
        print("CEventPointNet Read!")

    def Setting(self):
        print("CEventPointNet Setting!")

    def Reset(self):
        print("CEventPointNet Reset!")