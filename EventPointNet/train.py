import EventPointNet.nets as nets
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.Log import DebugPrint
import DBhandler.DBhandler as DBhandler
import numpy as np
import cv2
from common.utils import CudaStatus

class CTrain():
    def __init__(self, learningRate = 0.001):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate
        self.__sPrintStep = 100
        self.__batch_size = 1
        self.__strCkptDetPath = "./EventPointNet/checkpoints/checkpoint_detector.pth"
        self.__strCkptDescPath = "./EventPointNet/checkpoints/checkpoint_descriptor.pth"
        if(self.__device == "cuda"):
            self.__cudaStatus = CudaStatus()
        if(not os.path.exists(os.path.dirname(self.__strCkptDetPath))):
            os.mkdir(self.__strCkptDetPath)
        
    def Open(self, db, dbPath):
        self.__dbHandler = DBhandler.CDbHandler(db)
        self.__dbHandler.Open(dbPath)
        self.__train_loader = self.__dbHandler.Read(batch_size=self.__batch_size)
          
    def Setting(self, train_mode):
        if(train_mode == "train_keypt"):
            self.__model = nets.CDetectorNet().to(self.__device)
            self.__strCkptPath = self.__strCkptDetPath
        elif(train_mode == "train_desc"):
            self.__model = nets.CDescriptorNet().to(self.__device)
            self.__strCkptPath = self.__strCkptDescPath
        else:
            DebugPrint().error("Train mode error!")
            return 0
        self.__loss = torch.nn.MSELoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)
        if(os.path.isfile(self.__strCkptPath)):
            if(self.__device == "cuda"):
                self.__model.load_state_dict(torch.load(self.__strCkptPath))
            else:
                self.__model.load_state_dict(torch.load(self.__strCkptPath, map_location=torch.device("cpu")))
        return 1

    def train(self, train_mode):
        if(train_mode == "train_keypt"):
            self.train_keypt()
        elif(train_mode == "train_desc"):
            self.train_desc()

    def train_keypt(self):
        self.__model.train(True)
        
        sTrainIdx = 0
        for sBatch, data in enumerate(self.__train_loader):
            image, target = data['image'].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)

            for i in range(0, 2):
                self.__model.zero_grad()
                output, _ = self.__model.forward(image)
                output = output[:, :-1, :]
                output = torch.nn.functional.pixel_shuffle(output, 8)

                loss = self.__loss(output, target)
                loss.backward()
                self.__optimizer.step()
                if sTrainIdx % self.__sPrintStep == 0:
                    DebugPrint().info("Step: " + str(sTrainIdx) + ", Loss: " + str(loss.item()))
                    if(self.__device == "cuda"):
                        DebugPrint().info("Cuda Status: " + str(self.__cudaStatus["allocated"]) + "/" + str(self.__cudaStatus["total"]))
                sTrainIdx += 1

                image, target = data['rotimage'].to(self.__device, dtype=torch.float32), data['rottarget'].to(self.__device, dtype=torch.float32)
        torch.save(self.__model.state_dict(), self.__strCkptDetPath)

    def train_desc(self):
        self.__model.train(True)
        sTrainIdx = 0
        for sBatch, data in enumerate(self.__train_loader):
            image, target = data['image'].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)
            self.__model.zero_grad()
            output_dark = self.__model.forward(image)
            output_origin = self.__model.forward(target)
            
            loss = self.__loss(output_dark, output_origin)
            loss.backward()
            self.__optimizer.step()

            if sTrainIdx % self.__sPrintStep == 0:
                DebugPrint().info("Step: " + str(sTrainIdx) + ", Loss: " + str(loss.item()))
                if(self.__device == "cuda"):
                    DebugPrint().info("Cuda Status: " + str(self.__cudaStatus["allocated"]) + "/" + str(self.__cudaStatus["total"]))
            sTrainIdx += 1
        torch.save(self.__model.state_dict(), self.__strCkptDescPath)