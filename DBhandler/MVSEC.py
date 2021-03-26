import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2, os

class CMvsecDataset(Dataset):
    def __init__(self, dataPath, transforms=None):        
        self.__transforms = transforms
        self.__targetPath = dataPath + "/event/"
        self.__trainPath = dataPath + "/image/"
        self.__trainImageList = [os.path.basename(x) for x in glob(self.__trainPath + "*.png")]
        self.__targetList = [os.path.basename(x) for x in glob(self.__targetPath + "*.png")]
        self.__trainImageList.sort()
        self.__targetList.sort()

    def __len__(self):
        return len(self.__targetList)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = cv2.imread(self.__trainPath + self.__trainImageList[idx])
        target = cv2.imread(self.__targetPath + self.__targetList[idx])
        
        result = {'image': image, 'target': target}

        if self.__transforms:
            result = self.__transforms(result)

        return result

# event_dataset = CMvsecDataset(dataPath="/root/Workspace/sample")

# dataloader = DataLoader(event_dataset, batch_size=4, shuffle=True)
# for batchI, data in enumerate(dataloader):
#     print(batchI)
#     print(data['image'])
#     print(data['target'])