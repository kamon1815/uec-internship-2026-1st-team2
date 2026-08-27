#設定GUIを担当します
#settingGUI  = SettingGUI()
#で作成し、settingGUI.update()を呼び出すことで設定画面を表示します。
#settingGUI.update()の戻り値は{"camera":0, "contrast":1, "brightness":1, "auto":0},Falseのような辞書型と論理型です。
# "camera"の値が0でinfinicam、1でwebカメラを使用、"auto"の値が1でコントラストの自動調整ON、決定ボタンが押されたときだけ、二つ目の戻り値がTrueになります
# BattleクラスのchangeCameraConfigの引数に渡すことで設定を対戦画面へ引き継ぎます

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

    camera_prev_buttonstate = True

    def update(self):
        frame = np.zeros((800, 700, 3), np.uint8)
        frame[:] = (49, 52, 49)

        if cvui.button(frame, 550, 720, "設定完了"):#設定完了ボタンが押されたらウィンドウを閉じる
            self.is_acceptbutton_pressed = True
            self.close()
            return self.config,self.is_acceptbutton_pressed
        else:
            self.is_acceptbutton_pressed = False   

        if cvui.button(frame, 300, 720, "カメラ切り替え") and self.camera_prev_buttonstate:
            if self.config["camera"] == 1:
                self.config["camera"] = 0
            else:
                self.config["camera"] = 1

            self.camera_prev_buttonstate = False
        else:
            self.camera_prev_buttonstate = True

        #コントラストのトラックバー
        cvui.text(frame, 10, 540, 'コントラスト')
        cvui.trackbar(frame, 10, 560, 600, self.contrast,0.0, 2.0)
        #明るさのトラックバー
        cvui.text(frame, 10, 620, '明るさ')
        cvui.trackbar(frame, 10, 640, 600, self.brightness,-100, 100)
        #強調表示の切り替え

        cvui.checkbox(frame,10,10,'コントラスト強調', self.enhanceFilter)
        cvui.checkbox(frame,300,540,'コントラスト自動調整', self.autocontrast)

        self.config["contrast"] = self.contrast[0]
        self.config["brightness"] = self.brightness[0]
        self.config["auto"] = self.autocontrast[0]
        self.preview(frame)

        cvui.update()
        cvui.imshow(self.windowname, frame)



        return self.config,self.is_acceptbutton_pressed


    def preview(self,frame):
        img = self.getCameraImage()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)



        # コントラスト自動調整
        if self.autocontrast[0]:
            self.config["contrast"] = (255 / (max_val - min_val))
            self.contrast[0] = self.config["contrast"] 

        #プレビュー用の変換処理
        img = cv2.convertScaleAbs(img,alpha= self.config["contrast"],beta = self.config["brightness"])
        #グレースケール化
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #強調表示がオンの時はフィルタをかける
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


    def nothing(self,x):
        pass        

    def close(self):
        if self.__infinicam != None:
            self.__infinicam.close()
        self.__cap.release()
        cv2.waitKey(1)




    def __init__(self,):
        cvui.init(self.windowname)

        self.config = {"camera": 0, "contrast":1, "brightness":1,"auto":0}

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

            
        config,pressed = settingGUI.update()
        if pressed:
            print(config)
        key = cv2.waitKey(1)
        if settingGUI.is_acceptbutton_pressed or key == 27:
           break
    settingGUI.close()


