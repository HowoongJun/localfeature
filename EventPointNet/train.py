###
#
#       @Brief          train.py
#       @Details        Training code for EventPointNet
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 18, 2021
#       @Version        v0.13
#
###

import EventPointNet.nets as nets
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.Log import DebugPrint
import DBhandler.DBhandler as DBhandler
import numpy as np
import cv2, math
from common.utils import CudaStatus
from torch.utils.tensorboard import SummaryWriter

class CTrain():
    def __init__(self, learningRate = 0.00001):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate
        self.__sPrintStep = 100
        self.__batch_size = 1
        self.__strCkptPath = "./EventPointNet/checkpoints/checkpoint.pth"
        self.softmax = torch.nn.Softmax2d()
        if(self.__device == "cuda"):
            self.__cudaStatus = CudaStatus()
        if(not os.path.exists(os.path.dirname(self.__strCkptPath))):
            os.mkdir(self.__strCkptPath)
        
    def Open(self, db, dbPath):
        self.__dbHandler = DBhandler.CDbHandler(db)
        self.__dbHandler.Open(dbPath)
        self.__train_loader = self.__dbHandler.Read(batch_size=self.__batch_size)
          
    def Setting(self, train_mode):
        if(train_mode == "train"):
            self.__model = nets.CEventPointNet().to(self.__device)
            self.__loss = torch.nn.MSELoss()
        else:
            DebugPrint().error("Train mode error!")
            return 0
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)
        if(os.path.isfile(self.__strCkptPath)):
            if(self.__device == "cuda"):
                self.__model.load_state_dict(torch.load(self.__strCkptPath))
                DebugPrint().info("Checkpoint loaded")
            else:
                self.__model.load_state_dict(torch.load(self.__strCkptPath, map_location=torch.device("cpu")))
        return 1

    def train(self):
        self.__model.train(True)
        
        sTrainIdx = 0
        for sBatch, data in enumerate(self.__train_loader):
            image, target = data['image'].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)
            for i in range(0, 2):
                self.__model.zero_grad()
                kpt, desc = self.__model.forward(image)
                output = kpt[:, :-1, :]
                output = torch.nn.functional.pixel_shuffle(output, 8)
                target = target / 255.0

                descSift = self.__sift_descriptor(image)
                lossKpt = self.__loss(output, target)
                lossDsc = self.__loss(desc, descSift)
                loss = lossKpt + lossDsc
                loss.backward()
                self.__optimizer.step()
                if sTrainIdx % self.__sPrintStep == 0:
                    DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(loss.item()) + ", KptLoss: " + str(lossKpt.item()) + ", DscLoss: " + str(lossDsc.item()))
                    if(self.__device == "cuda"):
                        DebugPrint().info("Cuda Status: " + str(self.__cudaStatus["allocated"]) + "/" + str(self.__cudaStatus["total"]))
                sTrainIdx += 1

                image, target = data['rotimage'].to(self.__device, dtype=torch.float32), data['rottarget'].to(self.__device, dtype=torch.float32)
            if(sTrainIdx % 100 == 1):
                torch.save(self.__model.state_dict(), self.__strCkptPath)

    def __sift_descriptor(self, image):
        oSift = cv2.SIFT_create()
        oImage = np.squeeze(image.cpu().numpy(), axis=0).astype(np.uint8)
        _, height, width = oImage.shape
        kpt = []
        for h in range(0, height):
            for w in range(0, width):
                kpt.append(cv2.KeyPoint(h, w, 5.0))
        oImage = np.squeeze(oImage, axis=0)
        _, desc = oSift.compute(oImage, kpt)
        desc = torch.from_numpy(desc.reshape((1, 128, height, width))).to(self.__device)
        descNorm = torch.add(torch.norm(desc, p=2, dim=1), 0.00001)

        desc = desc.div(torch.unsqueeze(descNorm, 1))
        desc = desc * 2 - 1
        
        return desc


    # def train_desc(self):
    #     self.__model.train(True)
    #     sTrainIdx = 0
    #     oWriter = SummaryWriter()
    #     for sBatch, data in enumerate(self.__train_loader):
    #         target = data['target'].to(self.__device, dtype=torch.float32)
            
    #         for i in range(0, 5):
    #             self.__model.zero_grad()
    #             if(i < 4):
    #                 image = data['image0.' + str(i + 2)].to(self.__device, dtype=torch.float32)
    #             else:
    #                 image = data['target'].to(self.__device, dtype=torch.float32)
                
    #             output_dark = self.__model.forward(image)
    #             output_origin = self.__sift_descriptor(image)

    #             loss = self.__loss(output_dark, output_origin, torch.Tensor([1]).to(self.__device))
    #             if(math.isnan(loss.item())):
    #                 DebugPrint().warn("NaN!")
    #                 continue
    #             loss.backward()
    #             self.__optimizer.step()
    #             oWriter.add_scalar('loss_desc', loss, sTrainIdx)
    #             if sTrainIdx % self.__sPrintStep == 0:
    #                 DebugPrint().info("Step: " + str(sTrainIdx) + ", Loss: " + str(loss.item()))
    #                 if(self.__device == "cuda"):
    #                     DebugPrint().info("Cuda Status: " + str(self.__cudaStatus["allocated"]) + "/" + str(self.__cudaStatus["total"]))
    #             sTrainIdx += 1
    #     oWriter.close()
    #     torch.save(self.__model.state_dict(), self.__strCkptDescPath)