import EventPointNet.nets as nets
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common.Log import DebugPrint

class CTrain():
    def __init__(self, learningRate = 0.001):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate
        self.__sPrintStep = 100
    
    def run(self):
        self.__train(train_loader)
            
    def Setting(self):
        self.__model = nets.CEventPointNet().to(self.__device)
        self.__loss = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)

    def __train(self, train_loader):
        self.__model.train()
        sTrainIdx = 0
        for sBatch, (data, target) in enumerate(train_loader):
            data, target = data.to(self.__device), target.to(self.__device)
            self.__model.zero_grad()
            output, _ = self.__model.forward(data)
            
            loss = self.__loss(output, target)
            loss.backward()
            self.__optimizer.step()
            if sTrainIdx % self.__sPrintStep == 0:
                DebugPrint().info("Step: " + str(sTrainIdx) + ", Loss: " + str(loss))
            sTrainIdx += 1
            
        sTrainIdx = 0