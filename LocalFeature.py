###
#
#       @Brief          LocalFeature.py
#       @Details        Main class for visual localization local
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 3, 2021
#       @Version        v0.18.1
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
            self.__module = imp.load_source(model, "./model/superpoint/superpoint.py")
        elif model == "eventpointnet":
            log.DebugPrint().info("Model: EventPointNet")
            self.__module = imp.load_source(model, "./model/EventPointNet/eventpointnet.py")
        elif model == "eventpointnetpp":
            log.DebugPrint().info("Model: EventPointNet++")
            self.__module = imp.load_source(model, "./model/EventPointNetpp/eventpointnetpp.py")
        elif model == "orb":
            log.DebugPrint().info("Model: ORB")
            self.__module = imp.load_source(model, "./model/orb/orb.py")
        elif model == "sift":
            log.DebugPrint().info("Model: SIFT")
            self.__module = imp.load_source(model, "./model/sift/sift.py")
        elif model == "r2d2":
            log.DebugPrint().info("Model: R2D2")
            self.__module = imp.load_source(model, "./model/r2d2/r2d2.py")
        elif model == "keynet":
            log.DebugPrint().info("Model: Key.Net")
            self.__module = imp.load_source(model, "./model/keynet/keynet.py")
        elif model == "lift":
            log.DebugPrint().info("Model: LIFT")
            self.__module = imp.load_source(model, "./model/lift/lift.py")
        elif model == "akaze":
            log.DebugPrint().info("Model: AKAZE")
            self.__module = imp.load_source(model, "./model/akaze/akaze.py")
        elif model == "kaze":
            log.DebugPrint().info("Model: KAZE")
            self.__module = imp.load_source(model, "./model/kaze/kaze.py")
        elif model == "brisk":
            log.DebugPrint().info("Model: BRISK")
            self.__module = imp.load_source(model, "./model/brisk/brisk.py")
        elif model == "lfnet":
            log.DebugPrint().info("Model: LF-Net")
            self.__module = imp.load_source(model, "./model/lfnet/lfnet.py")

    def __del__(self):
        self.Close()

    def Open(self, argsmode, usegpu):
        self.__model = self.__module.CModel()
        if(usegpu == True and self.__gpuCheck == True): self.__gpuCheck = True
        elif(usegpu == False): self.__gpuCheck= False
        self.__model.Open(self.__gpuCheck, argsmode)
    
    def Close(self):
        self.__model.Close()

    def Read(self):
        return self.__model.Read()

    def Write(self, db, dbPath, train_mode="train"):
        self.__model.Write(db, dbPath, train_mode)

    def Setting(self, eCommand:int, Value=None):
        self.__model.Setting(eCommand, Value)

    def Reset(self):
        self.__model.Reset()
