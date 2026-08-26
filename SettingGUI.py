from InfinicamManeger import InfinicamManager
import numpy as np
import cv2
import time
import cvui

class SettingGUI:
    windowname = 'Setting'
    is_acceptbutton_pressed = False
    contrast = [1.0]
    brightness = [1.0]
    enhanceFilter = [False]
    autocontrast = [False]

    def update(self):
        frame = np.zeros((800, 700, 3), np.uint8)
        frame[:] = (49, 52, 49)

        if cvui.button(frame, 550, 720, "設定完了"):
            self.is_acceptbutton_pressed = True
            self.close()
            return
        else:
            self.is_acceptbutton_pressed = False   

        #コントラストのトラックバー
        cvui.text(frame, 10, 540, 'コントラスト')
        cvui.trackbar(frame, 10, 560, 600, self.contrast,0.0, 2.0)
        #明るさのトラックバー
        cvui.text(frame, 10, 620, '明るさ')
        cvui.trackbar(frame, 10, 640, 600, self.brightness,-30.0, 30)
        #強調表示の切り替え

        cvui.checkbox(frame,10,10,'コントラスト強調', self.enhanceFilter)
        cvui.checkbox(frame,300,540,'コントラスト自動調整', self.autocontrast)

        self.config["contrast"] = self.contrast[0]
        self.config["brightness"] = self.brightness[0]
   
        self.preview(frame)

        cvui.update()
        cvui.imshow(self.windowname, frame)

        return self.config


    def preview(self,frame):
        img = self.getCameraImage()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)



        # スケーリング係数の計算
        if self.autocontrast[0]:
            self.config["contrast"] = 255 / (max_val - min_val)    

        img = cv2.convertScaleAbs(img,alpha= self.config["contrast"],beta = self.config["brightness"])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        if self.enhanceFilter[0]:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            mappedimg = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            img = cv2.cvtColor(mappedimg, cv2.COLOR_BGR2RGB)
        
        cvui.image(frame, 10, 40,img)

    #現在のカメラ設定に応じてカメラから画像を取得する -------------------------------------------------------------------
    def getCameraImage(self):

        if self.config["camera"] == 0: # infinicamを使う場合
            if self.__infinicam != None: 
                img,defimg = self.__infinicam.get_frame()
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            else: 
                self.config["camera"] = 1 # infinicamが接続されていないならwebカメラを使うようにする

        if self.config["camera"] == 1: # webカメラを使う場合
            ret,img = self.__cap.read()

        return img


    #カメラ切り替えの機能 ---------------------------------------------------------------------------------------------
    def changeCamera(self,c):#c = 0でinfinicam、c = 1 でwebカメラ
        self.cameraMode = c


    def nothing(self,x):
        pass        

    def close(self):
        if self.__infinicam != None:
            self.__infinicam.close()
        self.__cap.release()
        cv2.waitKey(1)




    def __init__(self,):
        cvui.init(self.windowname)

        self.config = {"camera": 1, "contrast":1, "brightness":1}

                #infinicamとwebカメラをセットアップ
        try:
            self.__infinicam = InfinicamManager()
            self.__infinicam.connect(988,640,525,2.0,10) #fps , 解像度縦横, コントラスト, 明るさ
        except:
            self.__infinicam = None
            pass

        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 525)
        print(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



if __name__ == "__main__":

    settingGUI  = SettingGUI()

    while True:
            start_time = time.time()
            

            settingGUI.update()
            key = cv2.waitKey(1)
            if settingGUI.is_acceptbutton_pressed or key == 27:
               break
    settingGUI.close()


