# Testing code for local feature
from LocalFeature import *
from lcore.hal import eSettingCmd
import sys, os
from skimage import io, color
from skimage.transform import resize
import argparse
from glob import glob
import torch, cv2
import common.Log as log
import numpy as np

parser = argparse.ArgumentParser(description='Test Local Feature')
parser.add_argument('--model', '--m', type=str, default='mymodule', dest='model',
                    help='Model select: mymodule, superpoint, eventpointnet, sift, orb')
parser.add_argument('--resize', '--z', default='[1280,720]', dest='resize',
                    help='Resize image [width,height] (default = [1280,720]')
parser.add_argument('--channel', '--c', type=int, default=3, dest='channel',
                    help='Image channel (default = 3)')
parser.add_argument('--mode', '--o', type=str, dest='mode',
                    help='Mode select: makedb, query, match, train')
parser.add_argument('--query', '--q', type=str, dest='query',
                    help='Image query file path')
parser.add_argument('--match', '--a', type=str, dest='match',
                    help='Image match file path')
parser.add_argument('--thresh', '--t', type=int, default=3000, dest='threshold',
                    help='Threshold value for keypoint number')
parser.add_argument('--db', '--d', type=str, dest='db', default=None,
                    help='DB path for training')

args = parser.parse_args()

def imageRead(strImgPath):
    oImage = io.imread(strImgPath)
    if(oImage is None):
        return False
    if(len(oImage.shape) < 3):
        oImage = np.expand_dims(np.asarray(oImage), axis=0)
    elif(len(oImage.shape) == 3):
        oImage = (color.rgb2gray(oImage) * 255).astype(np.uint8)
        oImage = np.expand_dims(np.asarray(oImage), axis=0)
    return oImage

def readFolder(strImgFolder):
    if(not os.path.isdir(strImgFolder)):
        log.DebugPrint().warning("Path does not exist!")
        return False
    strPngList = [x for x in glob(strImgFolder + "*.png")]
    strJpgList = [x for x in glob(strImgFolder + "*.jpg")]
    strFileList = strPngList + strJpgList
    strFileList.sort()
    return strFileList

def queryCheck(oModel):
    if(args.query == None):
        log.DebugPrint().error("No query path")
        return False
    if(os.path.isdir(args.query)):
        strFileList = readFolder(args.query)
    else:
        strFileList = [args.query]
    if(strFileList is False):
        return False
    for fileIdx in strFileList:
        strImgPath = fileIdx
        oImage = imageRead(strImgPath)
        oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oImage)
        vKpt, vDesc = oModel.Read()
        oQuery = dict(image=oImage, keypoint=vKpt, descriptor=vDesc)
        oKptHandler = CKeypointHandler(args.mode, oQuery)
        oKptHandler.Save("./result/KptResult_" + str(args.model) + "_" + str(os.path.basename(fileIdx)))
        oKptHandler.Reset()
        oModel.Reset()
    return True

def checkGPU():
    if(torch.cuda.is_available()):
        log.DebugPrint().info("Using GPU.." + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        return True
    else:
        log.DebugPrint().info("Using CPU..")
        return False

def featureMatching(oModel):
    if(args.query == None or args.match == None):
        log.DebugPrint().error("No query / match path")
        return False
    if(os.path.isdir(args.query) or os.path.isdir(args.match)):
        log.DebugPrint().error("Query/match should be file, not folder")
        return False
    oImgQuery = imageRead(args.query)
    oImgMatch = imageRead(args.match)

    oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oImgQuery)
    vKptQuery, vDescQuery = oModel.Read()
    oQuery = dict(image=oImgQuery, keypoint=vKptQuery, descriptor=vDescQuery)
    oModel.Reset()
    oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, oImgMatch)
    vKptMatch, vDescMatch = oModel.Read()
    oMatch = dict(image=oImgMatch, keypoint=vKptMatch, descriptor=vDescMatch)
    oModel.Reset()

    oKptMatcher = CKeypointHandler(args.mode, oQuery, oMatch)
    oKptMatcher.Matching("bruteforce", args.model, ransac=100.0)
    strQueryName = os.path.splitext(os.path.basename(args.query))[0]
    oKptMatcher.Save("./result/MatchedResult_" + str(args.model) + str(strQueryName) + "_" + args.match)

if __name__ == "__main__":
    strModel = args.model

    model = CVisualLocLocal(strModel)
    model.Open(args.mode)
    model.Setting(eSettingCmd.eSettingCmd_IMAGE_CHANNEL, args.channel)
    if(args.mode == "makedb"):
        log.DebugPrint().info("[Local] DB Creation Mode")
    elif(args.mode == "query"):
        log.DebugPrint().info("[Local] Query Mode")
        model.Setting(eSettingCmd.eSettingCmd_THRESHOLD, args.threshold)
        queryCheck(model)
    elif(args.mode == "match"):
        log.DebugPrint().info("[Local] Matching Mode")
        model.Setting(eSettingCmd.eSettingCmd_THRESHOLD, args.threshold)
        featureMatching(model)
    elif(args.mode == "train"):
        log.DebugPrint().info("[Local] Train Mode")
        if(args.db == None):
            log.DebugPrint().error("[Local] No DB Path for Training!")
            sys.exit()
        model.Write("MVSEC", args.db)
    else:
        log.DebugPrint().error("[Local] Wrong mode! Please check the mode again")