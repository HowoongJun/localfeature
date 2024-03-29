###
#
#       @Brief          DBhandler.py
#       @Details        DB handler for training EventPointNet
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Apr. 01, 2021
#       @Version        v0.4
#
###


from common.Log import DebugPrint
from torch.utils.data import DataLoader
import os, imp

class CDbHandler():
    def __init__(self, db="MVSEC"):
        DebugPrint().info(db + " dataset loaded!")
        if(db == "MVSEC"):
            self.__db = imp.load_source(db, "./DBhandler/MVSEC.py")
        elif(db == "paris"):
            self.__db = imp.load_source(db, "./DBhandler/Paris.py")

    def Open(self, dbPath):
        if(not os.path.exists(dbPath)):
            DebugPrint().error("DB Path does not exist!")
            return False
        self.__Dataset = self.__db.CDataset(dataPath=dbPath)
        return True
    
    def Read(self, batch_size, shuffle=True):
        self.__dataLoader = DataLoader(self.__Dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return self.__dataLoader