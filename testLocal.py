# Testing code for local feature
from LocalFeature import *
from lcore.hal import eSettingCmd
import sys, os
from skimage import io
from skimage.transform import resize
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
parser.add_argument('--db', '--d', type=str, dest='db', default=None,
                    help='DB path for training')

args = parser.parse_args()

def imageRead(strImgPath):
    oImage = io.imread(strImgPath)
    if(oImage is None):
        return False
    if(len(oImage.shape) < 3):
        oImage = np.expand_dims(np.asarray(oImage), axis=0)
    # lResize = list(eval(args.resize))
    # iWidth = lResize[0]
    # iHeight = lResize[1]
    # if(args.channel == 1):
    #     oImage = cv2.cvtColor(oImage, cv2.COLOR_BGR2GRAY)
    # oImgResize = resize(oImage, (iWidth, iHeight))
    return oImage

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
        vKpt, vDesc = oModel.Read()
        
        heatmap = np.squeeze(vKpt, axis=0)
        heatmap = np.squeeze(heatmap, axis=0)
        xs, ys = np.where(heatmap >= 0.0154)
        a = imageRead(strImgPath)

        for k in range(0, len(xs)):
            a[0, xs[k], ys[k]] = 255
        # pts = np.zeros((3, len(xs)))
        # pts[0, :] = ys
        # pts[1, :] = xs
        # pts[2, :] = heatmap[xs, ys]
        a = np.squeeze(a, axis=0)
        io.imsave("./" + str(fileIdx) + ".png", a)
        # io.imsave("./" + str(fileIdx) + ".png", heatmap * 255)
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
    model.Open(args.mode)
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
        if(args.db == None):
            log.DebugPrint().error("[Local] No DB Path for Training!")
            sys.exit()
        model.Write("MVSEC", args.db)
    else:
        log.DebugPrint().error("[Local] Wrong mode! Please check the mode again")