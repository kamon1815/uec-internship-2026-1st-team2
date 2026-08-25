import cv2 # need to import extra module "pip install opencv-python"
import pypuclib
from pypuclib import CameraFactory, Camera, XferData, Decoder
from pypuclib import Resolution, PUCException, GPUSetup
import numpy as np

class InfinicamManager:
    def __init__(self):
        self.cam = None
        self.decoder = None
        self.reso = None
        self.GPUStatus = False
        self.width = None
        self.height = None
        self.alpha = None
        self.beta = None
        self.array = None
        self.decoded_deta = None
 #接続と初期設定
    def configurateCameraImage(self,contrast, light):
         self.alpha = contrast
         self.beta = light
    
    def connect(self, fps, width, height, alpha,beta):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
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
        print(f"{(width,height)}に設定しました") 
        print(f"{(alpha,beta)}に設定しました") 

#画像データの仲介,取得
    def get_frame(self):
        xferData = self.cam.grab()
        
            # Decode the data can be used as image
        if self.GPUStatus == True:
                self.decoded_deta = self.decoder.decodeGPU(xferData, True, self.reso.width)
        elif self.GPUStatus == False:
                self.decoded_deta = self.decoder.decode(xferData)
        if self.decoded_deta is None:
                return None, None  
        #変更前の状態をself.arrayに保存
        self.array = self.decoded_deta       
        #画面のリサイズ
        array = cv2.resize(self.array, (self.width, self.height)) 
        #明るさとコントラスト
        array = cv2.convertScaleAbs(array, alpha=self.alpha, beta=self.beta)
        #変更前と後をmainに渡す()
        return array, self.array    
#後片付け   
    def close(self):
        cv2.destroyAllWindows()
        if self.GPUStatus == True:
           self.decoder.teardownGPUDecode()

