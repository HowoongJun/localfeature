###
#
#       @Brief          train.py
#       @Details        Training code for EventPointNet
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 18, 2021
#       @Version        v0.14
#
###

import EventPointNet.nets as nets
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.Log import DebugPrint
import DBhandler.DBhandler as DBhandler
import numpy as np
import cv2, math, time
from common.utils import CudaStatus
from torch.utils.tensorboard import SummaryWriter

class CTrain():
    def __init__(self, learningRate = 0.0001):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate
        self.__sPrintStep = 10
        self.__batch_size = 1
        self.__strCkptPath = "./EventPointNet/checkpoints/checkpoint.pth"
        self.softmax = torch.nn.Softmax2d()
        if(not os.path.exists(os.path.dirname(self.__strCkptPath))):
            os.mkdir(self.__strCkptPath)
        
    def Open(self, db, dbPath):
        self.__dbHandler = DBhandler.CDbHandler(db)
        self.__dbHandler.Open(dbPath)
        self.__train_loader = self.__dbHandler.Read(batch_size=self.__batch_size)
        DebugPrint().info("Batch size: " + str(self.__batch_size))
          
    def Setting(self, train_mode):
        if(train_mode == "train" or train_mode == "reinforce"):
            self.__model = nets.CEventPointNet().to(self.__device)
            self.__lossKpt = torch.nn.MSELoss()
            self.__lossDsc = torch.nn.MSELoss()#torch.nn.CosineEmbeddingLoss()
        else:
            DebugPrint().error("Train mode error!")
            return 0
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)
        if(os.path.isfile(self.__strCkptPath)):
            if(self.__device == "cuda"):
                self.__model.load_state_dict(torch.load(self.__strCkptPath))
                DebugPrint().info("Checkpoint loaded at CUDA")
            else:
                self.__model.load_state_dict(torch.load(self.__strCkptPath, map_location=torch.device("cpu")))
                DebugPrint().info("Checkpoint loaded at CPU")
        return 1


    def train(self):
        dataList = ['image', 'bright0', 'bright1', 'bright2', 'rotimage', 'homimage']
        self.__model.train(True)
        oMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        sTrainIdx = 0
        self.__oSift = cv2.SIFT_create()
        for sBatch, data in enumerate(self.__train_loader):
            ckTime = time.time()
            checkdscloss = 0
            checkkptloss = 0
            for augment in dataList:
                if(augment == 'rotimage'):
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['rottarget'].to(self.__device, dtype=torch.float32)
                elif(augment == 'homimage'):
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['homtarget'].to(self.__device, dtype=torch.float32)
                else:
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)
                for augment2 in dataList:
                    if(augment >= augment2): continue
                    image2 = data[augment2].to(self.__device, dtype=torch.float32)
                    kpt, desc = self.__model.forward(image)
                    output = kpt[:, :-1, :]
                    output = torch.nn.functional.pixel_shuffle(output, 8)
                    target = target / 255.0
                    
                    kpt2, desc2 = self.__model.forward(image2)

                    vKpt1, vDesc1 = self.__GenerateLocalFeature(kpt, image)
                    vKpt2, vDesc2 = self.__GenerateLocalFeature(kpt2, image2)
                    
                    vMatches = oMatcher.match(vDesc1, vDesc2)
                    if(len(vMatches) < 4): continue
                    vKpSetQuery = np.float32([vKpt1[m.queryIdx].pt for m in vMatches]).reshape(-1, 1, 2)
                    vKpSetMatch = np.float32([vKpt2[m.trainIdx].pt for m in vMatches]).reshape(-1, 1, 2)
                    _, matchesMask = cv2.findHomography(vKpSetQuery, vKpSetMatch, cv2.RANSAC, 1.0)

                    vInlierSetQuery = [(point[0][0][0], point[0][0][1]) for point in vKpSetQuery[matchesMask]]
                    vInlierSetMatch = [(point[0][0][0], point[0][0][1]) for point in vKpSetMatch[matchesMask]]
                    
                    lossDsc = self.__lossDsc(torch.Tensor([1]), torch.Tensor([1])).to(self.__device)
                    for i in range(len(matchesMask)):
                        lossDsc += self.__lossDsc(desc[:,:,int(vInlierSetQuery[i][1]), int(vInlierSetQuery[i][0])], desc2[:,:,int(vInlierSetMatch[i][1]), int(vInlierSetMatch[i][0])])

                    lossKpt = self.__lossKpt(output, target)
                    checkdscloss += lossDsc.item()
                    checkkptloss += lossKpt.item()
                    loss = lossDsc + lossKpt
                    loss.backward()
                    self.__optimizer.step()
                    self.__optimizer.zero_grad()

            DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(checkdscloss / len(dataList) + checkkptloss / len(dataList)) + " = " + str(checkdscloss / len(dataList)) + " + " + str(checkkptloss / len(dataList)))
            DebugPrint().info("Time Consume: " + str(ckTime - time.time()))
            if(self.__device == "cuda"):
                cudaResult = CudaStatus()
                DebugPrint().info("Cuda Status: " + str(cudaResult["allocated"]) + "/" + str(cudaResult["total"]))
            if sTrainIdx % 10 == 0:
                torch.save(self.__model.state_dict(), self.__strCkptPath)
            sTrainIdx += 1
    
    def reinforce(self):
        sTrainIdx = 0
        self.__model.train(True)
        reinforceList = ['target', 'image0.5', 'image0.4', 'image0.3', 'image0.2']
        for sBatch, data in enumerate(self.__train_loader):
            for augment in reinforceList:
                image = data[augment].to(self.__device, dtype=torch.float32)
                for augment2 in reinforceList:
                    if(augment >= augment2): continue
                    image2 = data[augment2].to(self.__device, dtype=torch.float32)
                    kpt, desc = self.__model.forward(image)
                    kpt2, desc2 = self.__model.forward(image2)

                    vKpt1 = self.__GenerateLocalFeature(kpt)
                    vKpt2 = self.__GenerateLocalFeature(kpt2)
                    vKpt = vKpt1 + vKpt2
                    loss = self.__lossDsc(torch.Tensor([1]), torch.Tensor([1])).to(self.__device)

                    for i in range(len(vKpt)):
                        loss += self.__lossDsc(desc[:,:,vKpt[i][1], vKpt[i][0]], desc2[:,:,vKpt[i][1], vKpt[i][0]])
                    
                    # loss = self.__lossDsc(descConcat, desc2Concat)
                    loss.backward()
                    self.__optimizer.step()
                    if sTrainIdx % self.__sPrintStep == 0:
                        DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(loss.item()))
                        if(self.__device == "cuda"):
                            cudaResult = CudaStatus()
                            DebugPrint().info("Cuda Status: " + str(cudaResult["allocated"]) + "/" + str(cudaResult["total"]))
                    sTrainIdx += 1

                    self.__optimizer.zero_grad()

                # for i in range(1, 4):
                #     oImgSmall = data[augment]
                #     oImgSmall = oImgSmall[:, :, 0::2**i, 0::2**i].to(self.__device, dtype=torch.float32)
                #     _, desc = self.__model.forward(image)
                #     _, desc_small = self.__model.forward(oImgSmall)
                #     desc_small_target = desc[:, :, 0::2**i, 0::2**i]
                #     loss  = 1000 * self.__lossDsc(desc_small, desc_small_target, torch.Tensor([1,]).to(self.__device))
                #     loss.backward()
                #     self.__optimizer.step()
                #     if sTrainIdx % self.__sPrintStep == 0:
                #         DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(loss.item()))
                #         if(self.__device == "cuda"):
                #             cudaResult = CudaStatus()
                #             DebugPrint().info("Cuda Status: " + str(cudaResult["allocated"]) + "/" + str(cudaResult["total"]))
                #     sTrainIdx += 1

                #     self.__optimizer.zero_grad()
                torch.save(self.__model.state_dict(), self.__strCkptPath)

    def __GenerateLocalFeature(self, keypoint_distribution, image):
        kptDist = self.softmax(keypoint_distribution)
        kptDist = torch.exp(kptDist)
        kptDist = torch.div(kptDist, (torch.sum(kptDist[0], axis=0)+.00001))
        kptDist = kptDist[:,:-1,:]
        kptDist = torch.nn.functional.pixel_shuffle(kptDist, 8)
        kptDist = kptDist.data.cpu().numpy()

        heatmap = np.squeeze(kptDist, axis=0)
        heatmap = np.squeeze(heatmap, axis=0)
        heatmap_aligned = heatmap.reshape(-1)
        heatmap_aligned = np.sort(heatmap_aligned)[::-1]
        xs, ys = np.where(heatmap >= 0.015384)#heatmap_aligned[threshold])
        vKpt = []
        vDesc = []
        H, W = heatmap.shape
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.__Nms_fast(pts, H, W, 8)
        ys = pts[0, :]
        xs = pts[1, :]

        oImage = image.data.cpu().numpy()

        oImage = np.squeeze(oImage, axis=0)
        oImage = np.squeeze(oImage, axis=0).astype(np.uint8)

        for kptNo in range(len(xs)):
            # vKpt_tmp = [int(ys[kptNo]), int(xs[kptNo])]
            # vKpt.append(vKpt_tmp)

            vKpt_tmp = cv2.KeyPoint(int(ys[kptNo]), int(xs[kptNo]), 5.0)
            vKpt.append(vKpt_tmp)
        
        _, vDesc = self.__oSift.compute(oImage, vKpt)
        vDesc = np.array(vDesc)

        return vKpt, vDesc

    def __Nms_fast(self, in_corners, H, W, dist_thresh):
        grid = np.zeros((H, W)).astype(int) 
        inds = np.zeros((H, W)).astype(int) 
        
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int) 
        
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        
        pad = dist_thresh
        grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
        
        count = 0
        for i, rc in enumerate(rcorners.T):
        
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1:
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        
        keepy, keepx = np.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds