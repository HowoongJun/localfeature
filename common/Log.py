import logging, datetime
import os, colorlog

def DebugPrint(logger = None):
    oLogger = logging.getLogger(logger)
    if len(oLogger.handlers) > 0:
        return oLogger

    oLogger.setLevel(logging.DEBUG)
    dirname = "./log"
    if(not os.path.isdir(dirname)):
        os.mkdir(dirname)

    hFileHandler = logging.FileHandler(filename = dirname + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".log")
    oFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    hFileHandler.setFormatter(oFormatter)

    hConsole = logging.StreamHandler()
    hConsole.setLevel(logging.DEBUG)
    oColorFormat = colorlog.ColoredFormatter('%(log_color)s[%(levelname)s]%(name)s line %(lineno)s : %(message)s')
    hConsole.setFormatter(oColorFormat)

    oLogger.removeHandler(hConsole)
    oLogger.removeHandler(hFileHandler)
    oLogger.addHandler(hConsole)
    oLogger.addHandler(hFileHandler)

    return oLogger
