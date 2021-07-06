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
import cv2, math
from common.utils import CudaStatus
from torch.utils.tensorboard import SummaryWriter

class CTrain():
    def __init__(self, learningRate = 0.00001):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__learningRate = learningRate
        self.__sPrintStep = 10
        self.__batch_size = 1
        self.__strCkptPath = "./EventPointNet/checkpoints/checkpoint.pth"
        self.softmax = torch.nn.Softmax2d()
        if(self.__device == "cuda"):
            self.__cudaStatus = CudaStatus()
            DebugPrint().info("Device: Cuda")
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
            self.__lossDsc = torch.nn.CosineEmbeddingLoss()
        elif(train_mode == 'reinforce'):
            self.__model = nets.CEventPointNet().to(self.__device)
            self.__model_target = nets.CEventPointNet().to(self.__device)
            if(self.__device == "cuda"): self.__model_target.load_state_dict(torch.load(self.__strCkptPath))
            else: self.__model_target.load_state_dict(torch.load(self.__strCkptPath, map_location=torch.device("cpu")))
            self.__lossDsc = torch.nn.CosineEmbeddingLoss()
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
        dataList = ['bright0', 'bright1', 'bright2', 'rotimage']
        self.__model.train(True)
        sTrainIdx = 0
        for sBatch, data in enumerate(self.__train_loader):
            image, target = data['image'].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)
            # sift_image = data['bright2'].to(self.__device, dtype=torch.float32)
            for augment in dataList:
                self.__model.zero_grad()
                kpt, desc = self.__model.forward(image)
                output = kpt[:, :-1, :]
                output = torch.nn.functional.pixel_shuffle(output, 8)
                target = target / 255.0

                descSift = self.__sift_descriptor(image)
                lossDsc = 10.0 * self.__lossDsc(desc, descSift, torch.Tensor([1,]).to(self.__device))
                lossKpt = self.__lossKpt(output, target)
                loss = lossDsc + lossKpt
                loss.backward()
                self.__optimizer.step()
                if sTrainIdx % self.__sPrintStep == 0:
                    DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(loss.item()) + " = " + str(lossDsc.item()) + " + " + str(lossKpt.item()))
                    # if(self.__device == "cuda"):
                        # DebugPrint().info("Cuda Status: " + str(self.__cudaStatus["allocated"]) + "/" + str(self.__cudaStatus["total"]))
                sTrainIdx += 1
                if(augment == 'rotimage'): 
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['rottarget'].to(self.__device, dtype=torch.float32)
                    # sift_image = image
                else:
                    image, target = data[augment].to(self.__device, dtype=torch.float32), data['target'].to(self.__device, dtype=torch.float32)
                if(sTrainIdx % 100 == 0):
                    torch.save(self.__model.state_dict(), self.__strCkptPath)

    def __sift_descriptor(self, image):
        oSift = cv2.SIFT_create()
        desc_batch = []
        for uBatch in range(0, self.__batch_size):
            oImage = image[uBatch].cpu().numpy().astype(np.uint8)
            # oImage = np.squeeze(image.cpu().numpy(), axis=0).astype(np.uint8)
            _, height, width = oImage.shape
            kpt = []
            for h in range(0, height):
                for w in range(0, width):
                    kpt.append(cv2.KeyPoint(h, w, 5.0))
            oImage = np.squeeze(oImage, axis=0)
            _, desc = oSift.compute(oImage, kpt)
            desc = np.transpose(desc)
            desc = desc.reshape((128, height, width))
            descNorm = np.add(np.linalg.norm(desc), 0.00001)

            desc = desc / descNorm
            desc = desc * 2 - 1
            
            desc_batch.append(desc)
        desc_batch = np.array(desc_batch)
        desc_batch = torch.from_numpy(desc_batch).to(self.__device)
        return desc_batch
    
    def reinforce(self):
        sTrainIdx = 0
        reinforceList = ['target', 'image0.5', 'image0.4', 'image0.3', 'image0.2']
        for sBatch, data in enumerate(self.__train_loader):
            for augment in reinforceList:
                image = data[augment].to(self.__device, dtype=torch.float32)
                with torch.no_grad():
                    self.__model_target.to(self.__device)
                    self.__model_target.eval()
                    _, desc = self.__model_target.forward(image)
                    self.__model_target.cpu()
                for augment2 in reinforceList:
                    self.__model.train(True)
                    if(augment == augment2): continue
                    image2 = data[augment2].to(self.__device, dtype=torch.float32)
                    self.__model.zero_grad()
                    _, desc2 = self.__model.forward(image2)
                    
                    loss = 1000 * self.__lossDsc(desc, desc2, torch.Tensor([1,]).to(self.__device))
                    loss.backward()
                    self.__optimizer.step()
                    if sTrainIdx % self.__sPrintStep == 0:
                        DebugPrint().info("Step: " + str(sTrainIdx) + ", Total Loss: " + str(loss.item()))
                        # if(self.__device == "cuda"):
                            # DebugPrint().info("Cuda Status: " + str(self.__cudaStatus["allocated"]) + "/" + str(self.__cudaStatus["total"]))
                    sTrainIdx += 1
                    if(sTrainIdx % 100 == 0):
                        torch.save(self.__model.state_dict(), self.__strCkptPath)