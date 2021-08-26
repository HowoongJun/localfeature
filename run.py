###
#
#       @Brief          run.py
#       @Details        Testing code for local feature
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           May. 31, 2021
#       @Version        v0.13
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
from DBhandler.kitti import CKitti

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
    strPngList = [x for x in glob(strImgFolder + "/*.png")]
    strJpgList = [x for x in glob(strImgFolder + "/*.jpg")]
    strPpmList = [x for x in glob(strImgFolder + "/*.ppm")]
    strFileList = strPngList + strJpgList + strPpmList
    strFileList.sort()
    return strFileList

def queryCheck(oModel):
    if(args.query == None):
        log.DebugPrint().error("No query path")
        return False
    strFileList = []
    if(os.path.isdir(args.query)):
        strFolderList = [x[0] for x in os.walk(args.query)]
        for f in strFolderList:
            strFileList += readFolder(f)
    else:
        strFileList = [args.query]
    if(strFileList == []):
        log.DebugPrint().error("No File")
        return False
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    uCount = 0
    fTotalTime = 0
    for fileIdx in strFileList:
        vKpt, vDesc, oHeatmap, fTime = oEvaluation.Query(fileIdx, args.width, args.height)
        uCount += 1
        fTotalTime += fTime
        if(args.model == "eventpointnet" or args.model == "superpoint"):
            cv2.imwrite("./result/Heatmap_" + str(args.model) + "_" + str(os.path.basename(args.query)), oHeatmap)
    log.DebugPrint().info("========= Result of " + str(args.model) + "==========")
    log.DebugPrint().info("Total Time: " + str(fTotalTime))
    log.DebugPrint().info("Average Time: " + str(fTotalTime / uCount))

    return True

def featureMatching(oModel):
    if(args.query == None or args.match == None):
        log.DebugPrint().error("No query / match path")
        return False
    strQueryfileList = []
    strMatchfileList = []
    if(os.path.isdir(args.query) or os.path.isdir(args.match)):
        strQueryList = [x[0] for x in os.walk(args.query)]
        strMatchList = [x[0] for x in os.walk(args.match)]
        for f in strQueryList:
            strQueryfileList += [readFolder(f)]
        for f in strMatchList:
            strMatchfileList += [readFolder(f)]
        queryFiles = readFolder(args.query)
        matchFiles = readFolder(args.match)
    else:
        strQueryfileList = [[args.query]]
        strMatchfileList = [[args.match]]
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    for i in range(0, len(strQueryfileList)):
        for query in strQueryfileList[i]:
            for match in strMatchfileList[i]:
                if(args.query == args.match): 
                    if(query >= match): continue
                oHeatmapQuery, oHeatmapMatch = oEvaluation.Match(query, match, width=args.width, height=args.height, ransac=args.ransac)
                
                # if(args.model == "eventpointnet" or args.model == "superpoint"):
                #     if(oHeatmapQuery is not None):
                #         cv2.imwrite("./result/Heatmap_" + str(args.model) + "_" + str(os.path.basename(query)), oHeatmapQuery)
                #     if(oHeatmapMatch is not None):
                #         cv2.imwrite("./result/Heatmap_" + str(args.model) + "_" + str(os.path.basename(match)), oHeatmapMatch)

def slam(oModel):
    if(args.query == None):
        log.DebugPrint().error("No query / match path")
        return False
    if(os.path.isdir(args.query)):
        queryFiles = readFolder(args.query)
    else:
        queryFiles = [args.query]
    vResultPath = args.query.split(os.path.sep)
    oDraw = draw.CDraw(args.model, vResultPath[-3])
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    vTrans = np.array([0, 0, 0])
    vRot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    vPoseDraw = [vTrans.tolist()]
    oKitti = CKitti(vResultPath[-3])
    vGTPose = oKitti.getPose()
    vCalib = oKitti.getCalib()
    vRPE = []
    vATE = []
    for i in range(0,len(queryFiles) - 1, 1):
        log.DebugPrint().info(os.path.basename(queryFiles[i]) + " and " + os.path.basename(queryFiles[i + 1]))
        vPrevRot = vRot
        vPrevTrans = vTrans

        fGTScale = np.linalg.norm(vGTPose[i][:,3] - vGTPose[i+1][:, 3])
        vRot, vTrans = oEvaluation.SLAM(queryFiles[i], queryFiles[i + 1], vRot, vTrans, vCalib, scale = fGTScale, width=args.width, height=args.height, ransac=args.ransac)
        vTrans[1] = 0
        vPrevEstm = np.hstack((vPrevRot, np.expand_dims(vPrevTrans, axis=1)))
        vPrevEstm = np.vstack((vPrevEstm, [0, 0, 0, 1]))
        vEstm = np.hstack((vRot, np.expand_dims(vTrans, axis=1)))
        vEstm = np.vstack((vEstm, [0, 0, 0, 1]))
        
        mRPE = np.linalg.inv(np.linalg.inv(vGTPose[i]).dot(vGTPose[i+1])).dot(np.linalg.inv(vPrevEstm).dot(vEstm))
        mATE = np.linalg.inv(vGTPose[i+1]).dot(vEstm)
        vRPE.append(mRPE[:, 3])
        vATE.append(mATE[:, 3])
        vPoseDraw.append(vTrans.tolist())
        
        if(i % 100 == 0):
            oDraw.draw2D(vPoseDraw)
    RPE = np.sqrt(np.sum(np.linalg.norm(vRPE, axis=1)**2) / len(vRPE))
    ATE = np.sqrt(np.sum(np.linalg.norm(vATE, axis=1)**2) / len(vATE))

    log.DebugPrint().info("RPE: " + str(RPE))
    log.DebugPrint().info("ATE: " + str(ATE))
    oDraw.draw2D(vPoseDraw)

def hpatches(oModel):
    if(args.query == None):
        log.DebugPrint().error("Please write hpatches sequence folder in the --query argument")
        return False
    strHpatchesList = os.listdir(args.query)
    oEvaluation = CEvaluateLocalFeature(oModel, args.model)
    uCountIdx = 0
    uCount_i = 0
    uCount_v = 0
    fTotalMScore = 0
    fTotalRepeat = 0
    f_iMScore = 0
    f_vMScore = 0
    f_iRepeat = 0
    f_vRepeat = 0
    fRecTime = 0
    vResult = dict()
    vRepeatability = dict()
    for strHpatches in strHpatchesList:
        strHpatchesQueryList = readFolder(args.query + strHpatches + "/")
        query = args.query + strHpatches + "/1.ppm"
        fOneMScore = 0
        fOneRepeat = 0
        # for query in strHpatchesQueryList:
        for match in strHpatchesQueryList:
            if(query >= match): continue
            fMScore, fRepeatability = oEvaluation.HPatches(query, match, width=args.width, height=args.height, threshold=args.threshold, ransac=args.ransac)
            uCountIdx += 1
            fTotalMScore += fMScore
            fTotalRepeat += fRepeatability
            fOneMScore += fMScore
            fOneRepeat += fRepeatability
            if(strHpatches[0] == 'i'):
                uCount_i += 1
                f_iMScore += fMScore
                f_iRepeat += fRepeatability
            elif(strHpatches[0] == 'v'):
                uCount_v += 1
                f_vMScore += fMScore
                f_vRepeat += fRepeatability
        vRepeatability[strHpatches] = fOneRepeat / (len(strHpatchesQueryList) - 1)
        vResult[strHpatches] = fOneMScore / (len(strHpatchesQueryList) - 1)
        log.DebugPrint().info(vResult[strHpatches])
    np.save('./result/HPATCHES_' + str(args.model), vResult)
    np.save('./result/HPATCHES_REPEAT_' + str(args.model), vRepeatability)
    log.DebugPrint().info("================= Result of " + str(args.model) + "=========================")
    log.DebugPrint().info("Matching Score Total: " + str(fTotalMScore / uCountIdx))
    log.DebugPrint().info("Matching Score Illumination: " + str(f_iMScore / uCount_i))
    log.DebugPrint().info("Matching Score Viewpoint: " + str(f_vMScore / uCount_v))
    log.DebugPrint().info(" ******** ")
    log.DebugPrint().info("Repeatability Total: " + str(fTotalRepeat / uCountIdx))
    log.DebugPrint().info("Repeatability Illumination: " + str(f_iRepeat / uCount_i))
    log.DebugPrint().info("Repeatability Viewpoint: " + str(f_vRepeat / uCount_v))

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
    elif(args.mode == "hpatches"):
        log.DebugPrint().info("[Local] Hpatches Evaluation Mode")
        model.Setting(eSettingCmd.eSettingCmd_THRESHOLD, args.threshold)
        hpatches(model)
    else:
        log.DebugPrint().error("[Local] Wrong mode! Please check the mode again")