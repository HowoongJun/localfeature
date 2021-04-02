# Testing code for local feature
from LocalFeature import *
from lcore.hal import eSettingCmd
import sys, os, cv2
import argparse
from glob import glob
import torch
import common.Log as log
import numpy as np

parser = argparse.ArgumentParser(description='Test Local Feature')
parser.add_argument('--model', '--m', type=str, default='mymodule', dest='model',
                    help='Model select: mymodule, superpoint, eventpointnet')
parser.add_argument('--resize', '--z', default='[1280,720]', dest='resize',
                    help='Resize image [width,height] (default = [1280,720]')
parser.add_argument('--channel', '--c', type=int, default=3, dest='channel',
                    help='Image channel (default = 3)')
parser.add_argument('--mode', '--o', type=str, dest='mode',
                    help='Mode select: makedb, query, match, train')
parser.add_argument('--query', '--q', type=str, dest='query',
                    help='Image query file path')

args = parser.parse_args()

def imageRead(strImgPath):
    oImage = cv2.imread(strImgPath)
    if(oImage is None):
        return False
    lResize = list(eval(args.resize))
    iWidth = lResize[0]
    iHeight = lResize[1]
    if(args.channel == 1):
        oImage = cv2.cvtColor(oImage, cv2.COLOR_BGR2GRAY)
    oImgResize = cv2.resize(oImage, dsize=(iWidth, iHeight), interpolation = cv2.INTER_LINEAR)
    return oImgResize

def readFolder(strImgFolder):
    if(not os.path.isdir(strImgFolder)):
        log.DebugPrint().warning("Path does not exist!")
        return False
    strPngList = [os.path.basename(x) for x in glob(strImgFolder + "*.png")]
    strJpgList = [os.path.basename(x) for x in glob(strImgFolder + "*.jpg")]
    strFileList = strPngList + strJpgList
    strFileList.sort()
    return strFileList

def queryCheck(oModel):
    if(args.query == None):
        log.DebugPrint().error("No query path")
        return False
    strFileList = readFolder(args.query)
    if(strFileList is False):
        return False
    for fileIdx in strFileList:
        strImgPath = args.query + '/' + fileIdx
        oModel.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA, imageRead(strImgPath))
        vResult = oModel.Read()
        print(vResult)
        oModel.Reset()
    return True

def checkGPU():
    if(torch.cuda.is_available()):
        log.DebugPrint().info("Using GPU.." + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        return True
    else:
        log.DebugPrint().info("Using CPU..")
        return False

if __name__ == "__main__":
    strModel = args.model

    model = CVisualLocLocal(strModel)
    model.Open()
    model.Setting(eSettingCmd.eSettingCmd_IMAGE_CHANNEL, args.channel)
    # model.Setting(eSettingCmd.eSettingCmd_CONFIG, checkGPU())
    if(args.mode == "makedb"):
        log.DebugPrint().info("[Local] DB Creation Mode")
    elif(args.mode == "query"):
        log.DebugPrint().info("[Local] Query Mode")
        queryCheck(model)
    elif(args.mode == "match"):
        log.DebugPrint().info("[Local] Matching Mode")
    elif(args.mode == "train"):
        log.DebugPrint().info("[Local] Train Mode")
        model.Write("MVSEC", "/root/Workspace/dataset_eventcam/MVSEC/night/left/")
    else:
        log.DebugPrint().error("[Local] Wrong mode! Please check the mode again")
    # strImgPath = "./test.png"
    # img = cv2.imread(strImgPath)
    # if(img is None):
        # print("No Image!")
        # sys.quit()
    # iWidth = 1280
    # iHeight = 720
    # img = cv2.resize(img, dsize = (iWidth, iHeight), interpolation = cv2.INTER_LINEAR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # model.Control(img)
    # print(model.Read())