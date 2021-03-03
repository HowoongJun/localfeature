# Visual Localization core abstract class
from abc import *

class CVisualLocalizationCore(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        print("Visual Localization Core Constructor")
        
    @abstractmethod
    def __del__(self):
        print("Visual Localization Core Destructor")

    @abstractmethod
    def Open(self):
        print("Visual Localization Core Open")
    
    @abstractmethod
    def Close(self):
        print("Visual Localization Core Close")
    
    @abstractmethod
    def Write(self):
        print("Visual Localization Core Write")

    @abstractmethod
    def Read(self):
        print("Visual Localization Core Read")

    @abstractmethod
    def Control(self):
        print("Visual Localization Core Control")

    @abstractmethod
    def Reset(self):
        print("Visual Localization Core Reset")
        
