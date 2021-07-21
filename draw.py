from matplotlib import pyplot as plt
import numpy as np

class CDraw():
    def __init__(self, model_name):
        self.__strModelName = model_name
    
    def draw2D(self, data):
        data = np.array(data)
        
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        # plt.xlim([-300, 500])
        # plt.ylim([-700, 0])
        plt.plot(x, z)
        plt.savefig("./result/VO_" + self.__strModelName)
