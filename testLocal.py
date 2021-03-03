# Testing code for local feature
from LocalFeature import *
import sys
import cv2
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

if __name__ == "__main__":
    strArg = "mymodule"
    if len(sys.argv) > 1:
        strArg = sys.argv[1]

    model = CVisualLocLocal(strArg)
    model.Open()
    strImgPath = "./test.png"
    img = cv2.imread(strImgPath)
    if(img is None):
        print("No Image!")
        sys.quit()
    iWidth = 1280
    iHeight = 720
    img = cv2.resize(img, dsize = (iWidth, iHeight), interpolation = cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model.Control(img)
    print(model.Read())