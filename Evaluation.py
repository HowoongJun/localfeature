###
#
#       @Brief          Evaluation.py
#       @Details        Evaluation class for local feature
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Jun. 24, 2021
#       @Version        v0.6
#
###

import numpy as np
from common.Log import DebugPrint
from skimage import io, color, img_as_ubyte
from skimage.transform import resize
from lcore.hal import eSettingCmd
import os, cv2
import time, random

class CEvaluateLocalFeature():
    def __init__(self, model, model_name):
        self.__oModel = model
        self.__strModel = model_name
        self.__width_ratio = 1
        self.__height_ratio = 1
    
    def __ReadImage(self, image_path, width, height):
        oImage = io.imread(image_path)
        uWidth, uHeight, _ = oImage.shape
        if(oImage is None):
            return False
        if(width is not None or height is not None):
            oImage = resize(oImage, (height, width))
            self.__width_ratio = uWidth / width
            self.__height_ratio = uHeight / height
        oImageGray = (color.rgb2gray(oImage) * 255).astype(np.uint8)
        oImageGray = np.expand_dims(np.asarray(oImageGray), axis=0)
        oImage = np.expand_dims(img_as_ubyte(oImage), axis=0)
        # oImage = np.expand_dims((np.asarray(oImage) * 255).astype(np.uint8), axis=0)
        return oImage, oImageGray

    def Query(self, image_path, width = None, height = None):
        if(self.__oModel == None):
            DebugPrint().error("Model is None")
            return False
        oImage, oImageGray = self.__ReadImage(image_path, width, height)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oImageGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oImage)
        ckTime = time.time()
        vKpt, vDesc, oHeatmap = self.__oModel.Read()
        fSaveTime = time.time() - ckTime
        DebugPrint().info("Running Time: " + str(fSaveTime))
        DebugPrint().info("Keypoint number: " + str(len(vKpt)))
        oQuery = dict(image = oImageGray, keypoint = vKpt, descriptor = vDesc, colorimage = oImage)
        oKptHandler = CKeypointHandler("query", oQuery)
        vResultPath = image_path.split(os.path.sep)
        strResultPath = "./result/" + vResultPath[-3] + "/" + vResultPath[-2] + "/"
        if(not os.path.isdir(strResultPath)):
            os.makedirs(strResultPath)
        vImageName = os.path.splitext(os.path.basename(image_path))
        oKptHandler.Save(strResultPath + "/KptResult_" + vImageName[0] + "_" + self.__strModel + vImageName[1])
        
        cv2.imwrite(strResultPath + "/Heatmap_" + self.__strModel + "_" + str(vResultPath) + ".png", oHeatmap)

        oKptHandler.Reset()
        self.__oModel.Reset()
        return vKpt, vDesc, oHeatmap, fSaveTime

    def Match(self, query_path, match_path, width=None, height=None, ransac=100.0):
        if(self.__oModel == None):
            DebugPrint().error("Model is None")
            return False
        oQuery, oQueryGray = self.__ReadImage(query_path, width, height)
        oMatch, oMatchGray = self.__ReadImage(match_path, width, height)

        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oQueryGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oQuery)
        ckTime = time.time()
        vQueryKpt, vQueryDesc, oQueryHeatmap = self.__oModel.Read()
        DebugPrint().info("Running Time: " + str(time.time() - ckTime))
        DebugPrint().info("Query Keypt Number: " + str(len(vQueryKpt)))
        oQuery = dict(image = oQueryGray, keypoint=vQueryKpt, descriptor=vQueryDesc, colorimage = oQuery)
        self.__oModel.Reset()
        
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oMatchGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oMatch)
        ckTime = time.time()
        vMatchKpt, vMatchDesc, oMatchHeatmap = self.__oModel.Read()
        DebugPrint().info("Running Time: " + str(time.time() - ckTime))
        DebugPrint().info("Match Keypt Number: " + str(len(vMatchKpt)))
        oMatch = dict(image = oMatchGray, keypoint=vMatchKpt, descriptor=vMatchDesc, colorimage = oMatch)
        self.__oModel.Reset()
        
        oKptMatcher = CKeypointHandler("match", oQuery, oMatch)
        oKptMatcher.Matching("bruteforce", self.__strModel, ransac=ransac)
        
        vResultPath = query_path.split(os.path.sep)
        strResultPath = "./result/" + vResultPath[-3] + "/" + vResultPath[-2] + "/"
        if(not os.path.isdir(strResultPath)):
            os.makedirs(strResultPath)

        vMatchName = os.path.splitext(os.path.basename(match_path))
        strQueryName = os.path.splitext(os.path.basename(query_path))[0]
        # strMatchName = os.path.basename(match_path)
        oKptMatcher.Save(strResultPath + "/MatchedResult_" + str(strQueryName) + "_" + str(vMatchName[0]) + "_" + self.__strModel + str(vMatchName[1]))
        cv2.imwrite(strResultPath + "/Heatmap_" + self.__strModel + "_" + str(strQueryName) + ".png", oQueryHeatmap)
        cv2.imwrite(strResultPath + "/Heatmap_" + self.__strModel + "_" + str(vMatchName[0]) + ".png", oMatchHeatmap)
        

    def SLAM(self, query_path, match_path, prevR, prevT, calibration, scale = 0, width=None, height=None, ransac = 1.0):
        if(self.__oModel == None):
            DebugPrint().error("Model is None")
            return False

        oQuery, oQueryGray = self.__ReadImage(query_path, width, height)
        oMatch, oMatchGray = self.__ReadImage(match_path, width, height)

        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oQueryGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oQuery)
        ckTime = time.time()
        vQueryKpt, vQueryDesc, oQueryHeatmap = self.__oModel.Read()
        DebugPrint().info("Running Time: " + str(time.time() - ckTime))
        DebugPrint().info("Query Keypt Number: " + str(len(vQueryKpt)))
        oQuery = dict(image = oQueryGray, keypoint=vQueryKpt, descriptor=vQueryDesc)
        self.__oModel.Reset()
        # oMatchGray = (((oMatchGray / 255.0) ** (1.0 / 0.2)) * 255).astype(np.uint8)
        # oMatch = (((oMatch / 255.0) ** (1.0 / 0.2)) * 255).astype(np.uint8)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oMatchGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oMatch)
        ckTime = time.time()
        vMatchKpt, vMatchDesc, oMatchHeatmap = self.__oModel.Read()
        DebugPrint().info("Running Time: " + str(time.time() - ckTime))
        DebugPrint().info("Match Keypt Number: " + str(len(vMatchKpt)))
        oMatch = dict(image = oMatchGray, keypoint=vMatchKpt, descriptor=vMatchDesc)
        self.__oModel.Reset()
        
        oKptMatcher = CKeypointHandler("match", oQuery, oMatch)
        oKptMatcher.Matching("bruteforce", self.__strModel)
        vMatches = oKptMatcher.Read()

        vQueryKpt = np.int32([vQueryKpt[m.queryIdx].pt for m in vMatches])
        vMatchKpt = np.int32([vMatchKpt[m.trainIdx].pt for m in vMatches])
        
        E, mask = cv2.findEssentialMat(vQueryKpt, vMatchKpt, focal=float(calibration['P0'][0]), pp=(float(calibration['P0'][2]), float(calibration['P0'][6])), prob=0.999)
        # E, mask = cv2.findEssentialMat(vQueryKpt, vMatchKpt, focal=float(calibration['P0'][0]), pp=(float(calibration['P0'][2]), float(calibration['P0'][6])), method=cv2.RANSAC, threshold = ransac, prob=0.999)
        _, R, T, mask = cv2.recoverPose(E, vQueryKpt, vMatchKpt, focal=float(calibration['P0'][0]), pp=(float(calibration['P0'][2]), float(calibration['P0'][6])))
        if(scale != 0):
            vNormalizedT = T[:, 0] / np.linalg.norm(T[:, 0])
            vCurrentT = vNormalizedT * scale
        else:
            vCurrentT = T[:, 0]

        vRot, vTrans = prevR.dot(R), prevT + prevR.dot(vCurrentT)
        vTrans[1] = 0
        vPrevEstm = np.hstack((prevR, np.expand_dims(prevT, axis=1)))
        vPrevEstm = np.vstack((vPrevEstm, [0, 0, 0, 1]))
        vEstm = np.hstack((vRot, np.expand_dims(vTrans, axis=1)))
        vEstm = np.vstack((vEstm, [0, 0, 0, 1]))
        
        return vPrevEstm, vEstm
    
    def GetRPE(self, gt_pose, next_gt_pose, prev_estimation, curr_estimation):
        mRPE =  np.linalg.inv(np.linalg.inv(gt_pose).dot(next_gt_pose)).dot(np.linalg.inv(prev_estimation).dot(curr_estimation))
        return mRPE

    def GetATE(self, gt_pose, estimation):
        mATE = np.linalg.inv(gt_pose).dot(estimation)
        return mATE

    def HPatches(self, query_path, match_path, width = None, height = None, threshold=3000, ransac=100.0):
        if(self.__oModel == None):
            DebugPrint().error("Model is None")
            return False

        oQuery, oQueryGray = self.__ReadImage(query_path, width, height)
        oMatch, oMatchGray = self.__ReadImage(match_path, width, height)
        HomographyPath = os.path.dirname(query_path) + "/H_1_" + os.path.basename(match_path)[0]
        HomographyFile = open(HomographyPath, 'r')
        HomographyRead = HomographyFile.read()
        HomographyRead = HomographyRead.split()
        mHomography = np.array([[float(HomographyRead[0]), float(HomographyRead[1]), float(HomographyRead[2])],
                                [float(HomographyRead[3]), float(HomographyRead[4]), float(HomographyRead[5])],
                                [float(HomographyRead[6]), float(HomographyRead[7]), float(HomographyRead[8])]])

        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oQueryGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oQuery)
        ckTime = time.time()
        vQueryKpt, vQueryDesc, oQueryHeatmap = self.__oModel.Read()
        if(len(vQueryKpt) > threshold):
            vRandomIdx = random.sample(range(0, len(vQueryKpt)), threshold)
            vQueryKpt = np.array(vQueryKpt)
            vQueryDesc = np.array(vQueryDesc)
            vQueryKpt = vQueryKpt[vRandomIdx]
            vQueryDesc = vQueryDesc[vRandomIdx]
        fRecTime = time.time() - ckTime
        DebugPrint().info("Running Time: " + str(time.time() - ckTime))
        DebugPrint().info("Query Keypt Number: " + str(len(vQueryKpt)))
    
        vQueryDesc = self.__GenerateSIFT(vQueryKpt, oQueryGray)

        oQuery = dict(image = oQueryGray, keypoint=vQueryKpt, descriptor=vQueryDesc)
        self.__oModel.Reset()
        
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oMatchGray)
        self.__oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oMatch)
        ckTime = time.time()
        vMatchKpt, vMatchDesc, oMatchHeatmap = self.__oModel.Read()
        if(len(vMatchKpt) > threshold):
            vRandomIdx = random.sample(range(0, len(vMatchKpt)), threshold)
            vMatchKpt = np.array(vMatchKpt)
            vMatchDesc = np.array(vMatchDesc)
            vMatchKpt = vMatchKpt[vRandomIdx]
            vMatchDesc = vMatchDesc[vRandomIdx]
        DebugPrint().info("Running Time: " + str(time.time() - ckTime))
        DebugPrint().info("Match Keypt Number: " + str(len(vMatchKpt)))

        vMatchDesc = self.__GenerateSIFT(vMatchKpt, oMatchGray)

        oMatch = dict(image = oMatchGray, keypoint=vMatchKpt, descriptor=vMatchDesc)
        self.__oModel.Reset()
        
        uRepeatabilityCk = 0
        for i in range(0, len(vQueryKpt)):
            vKpt = np.array([vQueryKpt[i].pt[0] * self.__width_ratio, vQueryKpt[i].pt[1] * self.__height_ratio, 1])
            vCheckKpt = mHomography.dot(vKpt)
            vCheckKpt = np.array([vCheckKpt[0] / self.__width_ratio, vCheckKpt[1] / self.__height_ratio, 1])
            for j in range(0, len(vMatchKpt)):
                vKpt2 = np.array([vMatchKpt[j].pt[0], vMatchKpt[j].pt[1], 1])
                if(np.linalg.norm(vCheckKpt - vKpt2) <= 4): 
                    uRepeatabilityCk += 1
                    break
        if(min(len(vQueryKpt), len(vMatchKpt)) == 0):
            fRepeatability = 0
            fMScore = 0
        else:
            fRepeatability = uRepeatabilityCk / min(len(vQueryKpt), len(vMatchKpt))

            oKptMatcher = CKeypointHandler("match", oQuery, oMatch)
            oKptMatcher.Matching("bruteforce", self.__strModel, ransac=ransac)
            uMatchingNum = oKptMatcher.GetMatchingNumber()
        
            vMatches = oKptMatcher.Read()
            uCorrectMatches = 0
            for m in vMatches:
                vKpt = np.array([vQueryKpt[m.queryIdx].pt[0] * self.__width_ratio, vQueryKpt[m.queryIdx].pt[1] * self.__height_ratio, 1])
                vCheckKpt = mHomography.dot(vKpt)
                vCheckKpt = np.array([vCheckKpt[0] / self.__width_ratio, vCheckKpt[1] / self.__height_ratio, 1])
                vKpt2 = np.array([vMatchKpt[m.trainIdx].pt[0], vMatchKpt[m.trainIdx].pt[1], 1])
                if(np.linalg.norm(vKpt2 - vCheckKpt) <= 4): uCorrectMatches += 1
            fMScore = uCorrectMatches / uMatchingNum
        # fMScore = (uMatchingNum / len(vQueryKpt) + uMatchingNum / len(vMatchKpt)) / 2
        
        # strQueryName = os.path.splitext(os.path.basename(query_path))[0]
        # strMatchName = os.path.basename(match_path)
        # oKptMatcher.Save("./result/MatchedResult_" + str(self.__strModel) + str(strQueryName) + "_" + str(strMatchName))
        
        return fMScore, fRepeatability
    
    def __GenerateSIFT(self, kpt, image):
        oSift = cv2.SIFT_create()
        image = np.squeeze(image, axis=0)
        _, vDesc = oSift.compute(image, kpt)
        return vDesc

class CKeypointHandler():
    def __init__(self, mode, query, match=None):
        self.__mode = mode
        self.__vMatches = None
        self.__oQuery = query
        self.__oMatch = match
        self.__oImgMatch = None
        self.__matchesMask = None
        self.__uMatchingNumber = -1

    def Matching(self, matching, model, ransac=-1.0):
        oMatcher = None
        if(matching == "bruteforce"):
            if(model == "orb"):
                oMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                oMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        if(oMatcher == None):
            DebugPrint().error("Check matcher")
            return -1
        
        self.__vMatches = oMatcher.match(self.__oQuery['descriptor'], self.__oMatch['descriptor'])
        self.__matchesMask = None
        if(ransac != -1.0 and len(self.__vMatches) >= 4):
            vKpSetQuery = np.float32([self.__oQuery['keypoint'][m.queryIdx].pt for m in self.__vMatches]).reshape(-1, 1, 2)
            vKpSetMatch = np.float32([self.__oMatch['keypoint'][m.trainIdx].pt for m in self.__vMatches]).reshape(-1, 1, 2)
            _, self.__matchesMask = cv2.findHomography(vKpSetQuery, vKpSetMatch, cv2.RANSAC, ransac)
            DebugPrint().info("Matching Number (RANSAC): " + str(np.sum(self.__matchesMask)))
            self.__uMatchingNumber = np.sum(self.__matchesMask)
    
    def GetMatchingNumber(self):
        return self.__uMatchingNumber
    
    def Read(self):
       return self.__vMatches
        

    def Reset(self):
        self.__vMatches = None
        self.__matchesMask = None
        self.__oQuery = None
        self.__oMatch = None

    def Save(self, path):
        oImgResult = None
        if(self.__mode == "match"):
            if(len(self.__vMatches) == 0):
                DebugPrint().info("No matching points")
                return False
            # saveMatches = []
            # for a in range(0, len(self.__vMatches)):
            #     if(self.__matchesMask[a] == [1]):
            #         saveMatches.append(self.__vMatches[a])
            oImgResult = cv2.drawMatches(cv2.cvtColor(np.squeeze(self.__oQuery['colorimage'], axis=0), cv2.COLOR_RGB2BGR), 
                                        self.__oQuery['keypoint'], 
                                        cv2.cvtColor(np.squeeze(self.__oMatch['colorimage'], axis=0), cv2.COLOR_RGB2BGR), 
                                        self.__oMatch['keypoint'], 
                                        self.__vMatches, 
                                        # saveMatches,
                                        None, 
                                        matchColor=(0, 255, 0, 0),
                                        singlePointColor=(0, 0, 255, 0),
                                        matchesMask = self.__matchesMask,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        elif(self.__mode == "query"):
            if(len(self.__oQuery['keypoint']) == 0):
                DebugPrint().info("No matching points")
                return False

            oImgResult = cv2.drawKeypoints(cv2.cvtColor(np.squeeze(self.__oQuery['colorimage'], axis=0), cv2.COLOR_RGB2BGR),
                                          self.__oQuery['keypoint'],
                                          None,
                                          color=(0, 255, 0, 0))
        if(oImgResult is not None):
            cv2.imwrite(path, oImgResult)
            DebugPrint().info("Image Saved at " + str(path))

    # def SaveHeatmap(self, query_path, match_path = None):
    #     cv2.imwrite(query_path, self.__oQuery['heatmap'])
    #     if(match_path is not None):
    #         cv2.imwrite(match_path, self.__oMatch['heatmap'])