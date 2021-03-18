import EventPointNet.nets as nets
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CTrain():
    def __init__(self, learningRate = 0.001):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate

    def run(self, step):
        for i in range(step):
            self.__train(train_loader)
            
    def Setting(self):
        self.__model = nets.CBaseNetwork().to(self.__device)
        self.__loss = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learningRate)

    def __train(self, train_loader):
        self.__model.train()
        for data, target in enumerate(train_loader):
            data, target = data.to(self.__device), target.to(self.__device)
            self.__model.zero_grad()
            output, _ = self.__model(data)
            
            loss = self.__loss(output, target)
            loss.backward()
            self.__optimizer.step()