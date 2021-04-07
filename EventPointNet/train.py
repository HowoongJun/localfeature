import EventPointNet.nets as nets
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.Log import DebugPrint
import DBhandler.DBhandler as DBhandler
import numpy as np

class CTrain():
    def __init__(self, learningRate = 0.1):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate
        self.__sPrintStep = 100
        self.__batch_size = 4
        self.__strCkptPath = ".EventPointNet/checkpoints"
        if(not os.path.exists(self.__strCkptPath)):
            os.mkdir(self.__strCkptPath)
    
    def Open(self, db, dbPath):
        self.__dbHandler = DBhandler.CDbHandler(db)
        self.__dbHandler.Open(dbPath)
        self.__train_loader = self.__dbHandler.Read(batch_size=self.__batch_size)
       
    def run(self):
        self.__train()
    
    def Setting(self):
        self.__model = nets.CEventPointNet().to(self.__device)
        self.__loss = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)

    def __train(self):
        self.__model.train()
        
        sTrainIdx = 0
        for sBatch, data in enumerate(self.__train_loader):
            image, target = data['image'].to(self.__device, dtype=torch.float), data['target'].to(self.__device, dtype=torch.long)

            self.__model.zero_grad()
            output, _ = self.__model.forward(image)
            output = torch.reshape(output, (self.__batch_size, 1, 256, 344))
            ouptut = output.detach().cpu()
            
            target = target.squeeze(1)
            loss = self.__loss(output, target)
            loss.backward()
            
            self.__optimizer.step()
            if sTrainIdx % self.__sPrintStep == 0:
                DebugPrint().info("Step: " + str(sTrainIdx) + ", Loss: " + str(loss))
            sTrainIdx += 1
        sTrainIdx = 0
        torch.save(self.__model.state_dict(), self.__strCkptPath + "/checkpoint.pth")