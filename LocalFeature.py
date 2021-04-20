# Main class for visual localization local
import imp
from lcore.hal import *
import common.Log as log
import torch

class CVisualLocLocal(CVisualLocalizationCore):
    def __init__(self, model):
        if(torch.cuda.is_available()):
            self.__gpuCheck = True
        else:
            self.__gpuCheck = False

        if model == "mymodule":
            self.__module = imp.load_source(model, "./lcore/mymodule.py")
        elif model == "superpoint":
            print("Model: SuperPoint")
            self.__module = imp.load_source(model, "./localfeature_ref/superpoint/superpoint.py")
        elif model == "eventpointnet":
            log.DebugPrint().info("Model: EventPointNet")
            self.__module = imp.load_source(model, "./EventPointNet/eventpointnet.py")

    def __del__(self):
        self.Close()

    def Open(self, argsmode):
        self.__model = self.__module.CModel()
        self.__model.Open(self.__gpuCheck, argsmode)
    
    def Close(self):
        self.__model.Close()

    def Read(self):
        return self.__model.Read()

    def Write(self, db, dbPath):
        self.__model.Write(db, dbPath)

    def Setting(self, eCommand:int, Value=None):
        self.__model.Setting(eCommand, Value)

    def Reset(self):
        self.__model.Reset()
