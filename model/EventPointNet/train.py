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

import model.EventPointNet.nets as nets
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
        self.__strCkptPath = "./model/EventPointNet/checkpoints/checkpoint.pth"
        self.softmax = torch.nn.Softmax2d()
        if(not os.path.exists(os.path.dirname(self.__strCkptPath))):
            os.mkdir(self.__strCkptPath)
        
    def Open(self, db, dbPath):
        self.__dbHandler = DBhandler.CDbHandler(db)
        self.__dbHandler.Open(dbPath)
        self.__train_loader = self.__dbHandler.Read(batch_size=self.__batch_size)
        DebugPrint().info("Batch size: " + str(self.__batch_size))
          
    def Setting(self, train_mode):
        if(train_mode == "train"):
            self.__model = nets.CEventPointNet().to(self.__device)
            self.__lossKpt = torch.nn.MSELoss()
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
        dataList = ['image', 'bright0', 'bright1', 'bright2', 'dark0', 'dark1', 'dark2', 'rotimage']
        self.__model.train(True)
        sTrainIdx = 0
        for sBatch, data in enumerate(self.__train_loader):
            ckTime = time.time()
            checkkptloss = 0
            for augment in dataList:
                if(augment == 'rotimage'):
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['rottarget'].to(self.__device, dtype=torch.float32)
                else:
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)
                
                kpt = self.__model.forward(image)
                output = kpt[:, :-1, :]
                output = torch.nn.functional.pixel_shuffle(output, 8)
                target = target / 255.0

                loss = self.__lossKpt(output, target)
                
                loss.backward()
                self.__optimizer.step()
                self.__optimizer.zero_grad()

            DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(loss.item()))
            DebugPrint().info("Time Consume: " + str(ckTime - time.time()))
            if(self.__device == "cuda"):
                cudaResult = CudaStatus()
                DebugPrint().info("Cuda Status: " + str(cudaResult["allocated"]) + "/" + str(cudaResult["total"]))
            if sTrainIdx % 10 == 0:
                torch.save(self.__model.state_dict(), self.__strCkptPath)
            sTrainIdx += 1
