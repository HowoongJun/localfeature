
from matplotlib import pyplot as plt
import numpy as np

n = ['v', 'i']

models = ['eventpointnet', 'superpoint', 'sift', 'orb', 'brisk', 'kaze', 'akaze']
for k in n:
    eventpointnet  = np.load("./result/HPATCHES_eventpointnet.npy", allow_pickle=True)
    superpoint = np.load("./result/HPATCHES_superpoint.npy", allow_pickle=True)
    # r2d2 = np.load("./result/HPATCHES_r2d2.npy", allow_pickle=True)
    sift = np.load("./result/HPATCHES_sift.npy", allow_pickle=True)
    orb = np.load("./result/HPATCHES_orb.npy", allow_pickle=True)
    brisk = np.load("./result/HPATCHES_brisk.npy", allow_pickle=True)
    akaze = np.load("./result/HPATCHES_akaze.npy", allow_pickle=True)
    kaze = np.load("./result/HPATCHES_kaze.npy", allow_pickle=True)

    keys = list(eventpointnet.item().keys())
    dataKeys = []
    for i in keys:
        if(i[0] == k):
            dataKeys.append(i)

    eventpointnet_y = [eventpointnet.item()[i] for i in dataKeys]
    superpoint_y = [superpoint.item()[i] for i in dataKeys]
    sift_y = [sift.item()[i] for i in dataKeys]
    orb_y = [orb.item()[i] for i in dataKeys]
    brisk_y = [brisk.item()[i] for i in dataKeys]
    kaze_y = [kaze.item()[i] for i in dataKeys]
    akaze_y = [akaze.item()[i] for i in dataKeys]

    eventpointnet_x = [2*x for x in range(len(dataKeys))]
    superpoint_x = [2*x for x in range(len(dataKeys))]
    sift_x = [2*x for x in range(len(dataKeys))]
    orb_x =  [2*x for x in range(len(dataKeys))]
    brisk_x = [2*x for x in range(len(dataKeys))]
    kaze_x = [2*x for x in range(len(dataKeys))]
    akaze_x = [2*x  for x in range(len(dataKeys))]

    plt.bar(eventpointnet_x, eventpointnet_y, width=1.5)
    plt.bar(superpoint_x, superpoint_y, width=1.5)
    plt.bar(sift_x, sift_y, width=1.5)
    plt.bar(orb_x, orb_y, width=1.5)
    plt.bar(brisk_x, brisk_y, width=1.5)
    plt.bar(kaze_x, kaze_y, width=1.5)
    plt.bar(akaze_x, akaze_y, width=1.5)

    plt.legend(models)
    plt.xticks([2*i for i in range(0,len(dataKeys))], dataKeys, rotation=45, fontsize=2.5)
    if(k == 'i'):
        plt.savefig("./result/hpatches-sequences-release/plot_illumination.png", dpi=1000)
    elif(k == 'v'):
        plt.savefig("./result/hpatches-sequences-release/plot_viewpoint.png", dpi=1000)

    plt.figure()