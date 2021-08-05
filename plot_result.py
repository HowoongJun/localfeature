
from matplotlib import pyplot as plt
import numpy as np

n = ['v', 'i']
X_range = 16

models = ['eventpointnet', 'superpoint', 'sift', 'orb', 'brisk', 'kaze', 'akaze']

eventpointnet  = np.load("./result/HPATCHES_eventpointnet.npy", allow_pickle=True)
eventpointnet_repeat = np.load("./result/HPATCHES_REPEAT_eventpointnet.npy", allow_pickle=True)
superpoint = np.load("./result/HPATCHES_superpoint.npy", allow_pickle=True)
superpoint_repeat = np.load("./result/HPATCHES_REPEAT_superpoint.npy", allow_pickle=True)
# r2d2 = np.load("./result/HPATCHES_r2d2.npy", allow_pickle=True)
# sift = np.load("./result/HPATCHES_sift.npy", allow_pickle=True)
# orb = np.load("./result/HPATCHES_orb.npy", allow_pickle=True)
# brisk = np.load("./result/HPATCHES_brisk.npy", allow_pickle=True)
# akaze = np.load("./result/HPATCHES_akaze.npy", allow_pickle=True)
# kaze = np.load("./result/HPATCHES_kaze.npy", allow_pickle=True)

for k in n:
    keys = list(eventpointnet.item().keys())
    dataKeys = []
    for i in keys:
        if(i[0] == k):
            dataKeys.append(i)
    for repeat in range(0,2):
        if(repeat == 0):
            eventpointnet_y = [eventpointnet.item()[i] for i in dataKeys]
            superpoint_y = [superpoint.item()[i] for i in dataKeys]

        else:
            eventpointnet_y = [eventpointnet_repeat.item()[i] for i in dataKeys]
            superpoint_y = [superpoint_repeat.item()[i] for i in dataKeys]
        # sift_y = [sift.item()[i] for i in dataKeys]
        # orb_y = [orb.item()[i] for i in dataKeys]
        # brisk_y = [brisk.item()[i] for i in dataKeys]
        # kaze_y = [kaze.item()[i] for i in dataKeys]
        # akaze_y = [akaze.item()[i] for i in dataKeys]

        eventpointnet_x = [X_range*x for x in range(len(dataKeys))]
        superpoint_x = [X_range*x + 2 for x in range(len(dataKeys))]
        # sift_x = [X_range*x +4 for x in range(len(dataKeys))]
        # orb_x =  [X_range*x +6 for x in range(len(dataKeys))]
        # brisk_x = [X_range*x +8 for x in range(len(dataKeys))]
        # kaze_x = [X_range*x + 10 for x in range(len(dataKeys))]
        # akaze_x = [X_range*x + 12 for x in range(len(dataKeys))]

        plt.bar(eventpointnet_x, eventpointnet_y, width=1.5)
        plt.bar(superpoint_x, superpoint_y, width=1.5)
        # plt.bar(sift_x, sift_y, width=1.5)
        # plt.bar(orb_x, orb_y, width=1.5)
        # plt.bar(brisk_x, brisk_y, width=1.5)
        # plt.bar(kaze_x, kaze_y, width=1.5)
        # plt.bar(akaze_x, akaze_y, width=1.5)

        plt.legend(models)
        plt.xticks([X_range*i for i in range(0,len(dataKeys))], dataKeys, rotation=45, fontsize=2.5)
        if(k == 'i' and repeat == 0):
            plt.savefig("./result/hpatches-sequences-release/plot_mscore_illumination.png", dpi=1000)
        elif(k == 'v' and repeat == 0):
            plt.savefig("./result/hpatches-sequences-release/plot__mscore_viewpoint.png", dpi=1000)
        elif(k == 'i' and repeat == 1):
            plt.savefig("./result/hpatches-sequences-release/plot_repeat_illumination.png", dpi=1000)
        elif(k == 'v' and repeat == 1):
            plt.savefig("./result/hpatches-sequences-release/plot_repeat_viewpoint.png", dpi=1000)

        plt.figure()

