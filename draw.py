###
#
#       @Brief          draw.py
#       @Details        Drawing plots for visual odometry results
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Jul. 31, 2021
#       @Version        v0.3
#
###


from matplotlib import pyplot as plt
import numpy as np

class CDraw():
    def __init__(self, model_name, dataset_no):
        self.__strModelName = model_name
        self.__strDatasetNo = dataset_no
    
    def draw2D(self, data):
        data = np.array(data)
        
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        # plt.xlim([-300, 500])
        # plt.ylim([-700, 0])
        plt.plot(x, z)
        plt.savefig("./result/VO_" + self.__strDatasetNo + "_" + self.__strModelName)
        np.savez_compressed('./result/VO_' + self.__strDatasetNo + "_" + self.__strModelName, x=x, y=y, z=z)
