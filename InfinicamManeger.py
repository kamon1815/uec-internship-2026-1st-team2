import cv2 # need to import extra module "pip install opencv-python"
import pypuclib
from pypuclib import CameraFactory, Camera, XferData, Decoder
from pypuclib import Resolution, PUCException, GPUSetup
from pathlib import Path


class InfinicamManager:
    def __init__(self):
        self.cam = None
        self.decoder = None
        self.reso = None
        self.GPUStatus = False
 #接続と初期設定
    def connect(self, fps):
        self.cam = CameraFactory().create()
        self.decoder = self.cam.decoder()
        self.reso = self.cam.resolution()
        self.cam.setFramerateShutter(fps, fps)
        print(f"{fps}に設定しました")
        self.GPUStatus = self.decoder.getAvailableGPUProcess()
        if self.GPUStatus == True:
            param = GPUSetup(self.reso.width, self.reso.height)
            self.decoder.setupGPUDecode(param)
            print("Decode using a GPU device")
        elif self.GPUStatus == False:
            print("Since GPU is not available, decode using CPU")    
#画像データの仲介,取得
    def get_frame(self):
        xferData = self.cam.grab()
        
            # Decode the data can be used as image
        if self.GPUStatus == True:
                array = self.decoder.decodeGPU(xferData, True, self.reso.width)
        elif self.GPUStatus == False:
                array = self.decoder.decode(xferData)
        return array
#後片付け   
    def close(self):
        cv2.destroyAllWindows()
        if self.GPUStatus == True:
           self.decoder.teardownGPUDecode()
