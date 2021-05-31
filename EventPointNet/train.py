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
        self.__batch_size = 4
        self.__strCkptPath = "./EventPointNet/checkpoints/checkpoint.pth"
        if(not os.path.exists(os.path.dirname(self.__strCkptPath))):
            os.mkdir(self.__strCkptPath)
        
        
    def Open(self, db, dbPath):
        self.__dbHandler = DBhandler.CDbHandler(db)
        self.__dbHandler.Open(dbPath)
        self.__train_loader = self.__dbHandler.Read(batch_size=self.__batch_size)
       
    def run(self):
        self.__train()
    
    def Setting(self):
        self.__model = nets.CEventPointNet().to(self.__device)
        self.__loss = torch.nn.MSELoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)
        if(os.path.isfile(self.__strCkptPath)):
            if(self.__device == "cuda"):
                self.__model.load_state_dict(torch.load(self.__strCkptPath))
                self.__cudaStatus = CudaStatus()
            else:
                self.__model.load_state_dict(torch.load(self.__strCkptPath, map_location=torch.device("cpu")))

    def __train(self):
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
        torch.save(self.__model.state_dict(), self.__strCkptPath)