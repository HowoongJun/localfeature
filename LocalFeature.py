# Main class for visual localization local
import imp
import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
from lcore.hal import *
import common.Log as log
deprecation._PRINT_DEPRECATION_WARNINGS = False

class CVisualLocLocal(CVisualLocalizationCore):
    def __init__(self, model):
        if(tf.config.experimental.list_physical_devices('GPU')):
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

    def Open(self):
        self.__model = self.__module.CModel()
        self.__model.Open(self.__gpuCheck)
    
    def Close(self):
        self.__model.Close()

    def Read(self):
        return self.__model.Read()

    def Write(self):
        self.__model.Write()

    def Setting(self, oImage):
        self.__image = oImage
        self.__model.Control(oImage = self.__image)

    def Reset(self):
        self.__model.Reset()
