# Main class for visual localization local
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

class CKeypointHandler():
    def __init__(self, mode, query, match=None):
        self.__mode = mode
        self.__vMatches = None
        self.__oQuery = query
        self.__oMatch = match
        self.__oImgMatch = None
        self.__matchesMask = None

    def Matching(self, matching, model, ransac=-1):
        oMatcher = None
        if(matching == "bruteforce"):
            if(model == "orb"):
                oMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                oMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        if(oMatcher == None):
            log.DebugPrint().error("Check matcher")
            return -1
        
        self.__vMatches = oMatcher.match(self.__oQuery['descriptor'], self.__oMatch['descriptor'])
        self.__matchesMask = None        
        if(ransac != -1.0 and len(self.__vMatches) >= 4):
            vKpSetQuery = np.float32([self.__oQuery['keypoint'][m.queryIdx].pt for m in self.__vMatches]).reshape(-1, 1, 2)
            vKpSetMatch = np.float32([self.__oMatch['keypoint'][m.trainIdx].pt for m in self.__vMatches]).reshape(-1, 1, 2)
            _, self.__matchesMask = cv2.findHomography(vKpSetQuery, vKpSetMatch, cv2.RANSAC, ransac)
            log.DebugPrint().info("Matching Number (RANSAC): " + str(np.sum(self.__matchesMask)))

    def Reset(self):
        self.__vMatches = None
        self.__matchesMask = None
        self.__oQuery = None
        self.__oMatch = None

    def Save(self, path):
        oImgResult = None
        if(self.__mode == "match"):
            if(len(self.__vMatches) == 0):
                log.DebugPrint().info("No matching points")
                return False
            oImgResult = cv2.drawMatches(np.squeeze(self.__oQuery['image'], axis=0), 
                                        self.__oQuery['keypoint'], 
                                        np.squeeze(self.__oMatch['image'], axis=0), 
                                        self.__oMatch['keypoint'], 
                                        self.__vMatches, 
                                        None, 
                                        matchColor=(0, 255, 0, 0),
                                        singlePointColor=(0, 0, 255, 0),
                                        matchesMask = self.__matchesMask,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        elif(self.__mode == "query"):
            if(len(self.__oQuery['keypoint']) == 0):
                log.DebugPrint().info("No matching points")
                return False

            oImgResult = cv2.drawKeypoints(np.squeeze(self.__oQuery['image'], axis=0),
                                          self.__oQuery['keypoint'],
                                          None,
                                          color=(0, 255, 0, 0))
        if(oImgResult is not None):
            cv2.imwrite(path, oImgResult)
            log.DebugPrint().info("Image Saved at " + str(path))

    # def SaveHeatmap(self, query_path, match_path = None):
    #     cv2.imwrite(query_path, self.__oQuery['heatmap'])
    #     if(match_path is not None):
    #         cv2.imwrite(match_path, self.__oMatch['heatmap'])