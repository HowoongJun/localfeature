###
#
#       @Brief          LocalFeature.py
#       @Details        Main class for visual localization local
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 3, 2021
#       @Version        v0.16
#
###

import imp
from lcore.hal import *
import common.Log as log
import torch, cv2
import numpy as np

class CVisualLocLocal(CVisualLocalizationCore):
    def __init__(self, model):
        if(torch.cuda.is_available()):
            self.__gpuCheck = True
        else:
            self.__gpuCheck = False

        if model == "superpoint":
            log.DebugPrint().info("Model: SuperPoint")
            self.__module = imp.load_source(model, "./localfeature_ref/superpoint/superpoint.py")
        elif model == "eventpointnet":
            log.DebugPrint().info("Model: EventPointNet")
            self.__module = imp.load_source(model, "./EventPointNet/eventpointnet.py")
        elif model == "orb":
            log.DebugPrint().info("Model: ORB")
            self.__module = imp.load_source(model, "./localfeature_ref/orb/orb.py")
        elif model == "sift":
            log.DebugPrint().info("Model: SIFT")
            self.__module = imp.load_source(model, "./localfeature_ref/sift/sift.py")
        elif model == "r2d2":
            log.DebugPrint().info("Model: R2D2")
            self.__module = imp.load_source(model, "./localfeature_ref/r2d2/r2d2.py")
        elif model == "keynet":
            log.DebugPrint().info("Model: Key.Net")
            self.__module = imp.load_source(model, "./localfeature_ref/keynet/keynet.py")

    def __del__(self):
        self.Close()

    def Open(self, argsmode):
        self.__model = self.__module.CModel()
        self.__model.Open(self.__gpuCheck, argsmode)
    
    def Close(self):
        self.__model.Close()

    def Read(self):
        return self.__model.Read()

    def Write(self, db, dbPath, train_mode="train_keypt"):
        self.__model.Write(db, dbPath, train_mode)

    def Setting(self, eCommand:int, Value=None):
        self.__model.Setting(eCommand, Value)

    def Reset(self):
        self.__model.Reset()
