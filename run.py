###
#
#       @Brief          run.py
#       @Details        Testing code for local feature
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           May. 31, 2021
#       @Version        v0.11
#
###

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
import time
from Evaluation import *
import draw

parser = argparse.ArgumentParser(description='Test Local Feature')
parser.add_argument('--model', '-m', type=str, default='eventpointnet', dest='model',
                    help='Model select: superpoint, eventpointnet, lfnet, sift, orb, akaze, kaze, brisk, r2d2')
parser.add_argument('--width', '-W', type=int, default=None, dest='width',
                    help='Width for resize image')
parser.add_argument('--height', '-H', type=int, default=None, dest='height',
                    help='Height for resize image')
parser.add_argument('--channel', '-c', type=int, default=3, dest='channel',
                    help='Image channel (default = 3)')
parser.add_argument('--mode', '-o', type=str, dest='mode',
                    help='Mode select: makedb, query, match, train')
parser.add_argument('--query', '-q', type=str, dest='query',
                    help='Image query file path')
parser.add_argument('--match', '-a', type=str, dest='match',
                    help='Image match file path')
parser.add_argument('--thresh', '-t', type=int, default=3000, dest='threshold',
                    help='Threshold value for keypoint number')
parser.add_argument('--db', '-d', type=str, dest='db', default=None,
                    help='DB path for training')
parser.add_argument('--ransac', '-r', type=float, default=100.0, dest='ransac',
                    help='RANSAC Threshold value')
parser.add_argument('--gpu', '-g', type=int, default=1, dest='gpu',
                    help='Use GPU (1/0)')

args = parser.parse_args()

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
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    for fileIdx in strFileList:
        vKpt, vDesc, oHeatmap = oEvaluation.Query(fileIdx, args.width, args.height)
        
        if(args.model == "eventpointnet" or args.model == "superpoint"):
            cv2.imwrite("./result/Heatmap_" + str(args.model) + "_" + str(os.path.basename(args.query)), oHeatmap)
    return True

def featureMatching(oModel):
    if(args.query == None or args.match == None):
        log.DebugPrint().error("No query / match path")
        return False
    if(os.path.isdir(args.query) or os.path.isdir(args.match)):
        queryFiles = readFolder(args.query)
        matchFiles = readFolder(args.match)
    else:
        queryFiles = [args.query]
        matchFiles = [args.match]
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    for query in queryFiles:
        for match in matchFiles:
            if(query == match): continue
            oHeatmapQuery, oHeatmapMatch = oEvaluation.Match(query, match, width=args.width, height=args.height, ransac=args.ransac)
            
            if(args.model == "eventpointnet" or args.model == "superpoint"):
                if(oHeatmapQuery is not None):
                    cv2.imwrite("./result/Heatmap_" + str(args.model) + "_" + str(os.path.basename(query)), oHeatmapQuery)
                if(oHeatmapMatch is not None):
                    cv2.imwrite("./result/Heatmap_" + str(args.model) + "_" + str(os.path.basename(match)), oHeatmapMatch)

def slam(oModel):
    if(args.query == None):
        log.DebugPrint().error("No query / match path")
        return False
    if(os.path.isdir(args.query)):
        queryFiles = readFolder(args.query)
    else:
        queryFiles = [args.query]
    oDraw = draw.CDraw(args.model)
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    vTrans = np.array([0, 0, 0])
    vRot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vPoseDraw = [vTrans.tolist()]
    for i in range(0,len(queryFiles) - 1):
        log.DebugPrint().info(os.path.basename(queryFiles[i]) + " and " + os.path.basename(queryFiles[i + 1]))
        imageQuery = cv2.imread(queryFiles[i])
        
        vRot, vTrans = oEvaluation.SLAM(queryFiles[i], queryFiles[i+1], vRot, vTrans, width=args.width, height=args.height, ransac=args.ransac)
        vPoseDraw.append(vTrans.tolist())
        
        if(i % 100 == 0):
            oDraw.draw2D(vPoseDraw)
    oDraw.draw2D(vPoseDraw)

if __name__ == "__main__":
    strModel = args.model

    model = CVisualLocLocal(strModel)
    model.Open(args.mode, args.gpu)
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
        model.Write("MVSEC", args.db, args.mode)
    elif(args.mode == "reinforce"):
        log.DebugPrint().info("[Local] Reinforce Mode")
        if(args.db == None):
            log.DebugPrint().error("[Local] No DB Path for Reinforcing!")
            sys.exit()
        model.Write("paris", args.db, args.mode)
    elif(args.mode == "slam"):
        log.DebugPrint().info("[Local] SLAM Mode")
        slam(model)
    else:
        log.DebugPrint().error("[Local] Wrong mode! Please check the mode again")