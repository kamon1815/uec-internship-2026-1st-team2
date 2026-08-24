#試合関係を処理します
#コード上の方にある,cameraMode = 1 を cameraMode = 0 にするとinfinicamで動作確認ができます
#
#playOneMatch内のcomputeRPS(40,2,100)の引数は、撮影失敗時に取り直す回数、最終的に出力する枚数、その撮影間隔となっています。
#この場合は、100ミリ秒間隔で2回の判定を行い、撮影失敗時はそれぞれ最大40回まで再撮影します
#戻り値は   確率(辞書型),ジェスチャー(文字列)の順です。詳しくはJugdeHandをご確認ください。
#
#Battle.pyを実行するとCUIでのじゃんけんの確認ができます。ESCキーで終了します。

from FingerTracking import Fingertracking
from InfinicamManeger import InfinicamManager
from JugdeHand import JugdeHand
import numpy as np
import cv2
import time

class Battle:

    cameraMode = 1 # 0: inficicamを使用する 1: webカメラを使用する


    def __init__(self):

        if self.cameraMode == 0:
            self.__infinicam = InfinicamManager()
            self.__infinicam.connect(500,1246,1024,2.0,1) #fps , 解像度縦横, コントラスト, 明るさ

        if self.cameraMode == 1:
            self.__cap = cv2.VideoCapture(0)

        self.__tracker = Fingertracking()

        self.__judge = JugdeHand()


    def getCameraImage(self):

        if self.cameraMode == 0:
            img,defimg = self.__infinicam.get_frame()
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        if self.cameraMode == 1:
            ret,img = self.__cap.read()
        return img


    def computeRPS(self, maxtrial = 40,amount = 2, duration = 100 ): # 画像取得からパーセンテージの計算までを行います   maxtrialは撮影失敗時に取り直す回数、 amountは最終的に出力する枚数、durationは撮影間隔

        rsp_rates = []
        rsp_gestures = []
        for i in range(amount):
            for trial in range(maxtrial): #i回目の画像取得。失敗ならmaxtrial回まで再試行。

                img = self.getCameraImage()

                self.__tracker.doTracking(img,debug=True,useColor=False)
                
                if self.__tracker.getNormalizedPosition().get(0) != None: #トラッキング成功時はそれを判定へ回す
                    break
 
            #i回目の画像について判定する
            rate,gesture = self.__judge.finPos2rpsRate(self.__tracker)
            rsp_rates.append(rate)
            rsp_gestures.append(gesture)
            time.sleep(duration /1000)

        return rsp_rates,rsp_gestures


    def showRpsRate(self, rpsRate, name):

        img = np.zeros((300,500,3), dtype='uint8')
        img[:] = (255,255,255)
        if rpsRate['rock'] is not None: 
            img[10:90 ,0:int(rpsRate['rock']*500)] = (0,0,255)
        if rpsRate['rock'] is not None: 
            img[110:190 ,0:int(rpsRate['scissor']*500)] = (0,255,0)
        if rpsRate['rock'] is not None: 
            img[210:290 ,0:int(rpsRate['paper']*500)] = (255,0,0)
        cv2.imshow(name, img)


    def playOneMatch(self,timeCounter,oneshotFlags):# 1試合分の一連の処理を行います ----------------------------------------------------------------------------------------------------------------------------------

        ##カメラからの映像を常に表示
        cv2.imshow("Camera Stream",self.getCameraImage())

        if 0 < timeCounter <= 1:
            if(oneshotFlags[0]):
                print("最初は")
                oneshotFlags[0] = False
        if 1 < timeCounter <= 2:
            if(oneshotFlags[1]):
                print("グー")
                oneshotFlags[1] = False

        if 2 < timeCounter <= 2.5:
            if(oneshotFlags[2]):
                print("じゃん")
                oneshotFlags[2] = False
        if 2.5 <= timeCounter <= 2.9:
            if(oneshotFlags[3]):
                print("けん")
                oneshotFlags[3] = False
        if 2.9 <= timeCounter <= 4:
            if(oneshotFlags[4]):
                print("ぽん")

                #ここで骨格認識と判定
                rpsrates,gestures = self.computeRPS(40,2,200)
                print(rpsrates)

                #デバッグ用改造済みshowRpsRate
                self.showRpsRate(rpsrates[0],"1")
                self.showRpsRate(rpsrates[1],"2")
                oneshotFlags[4] = False

   
        if 7 < timeCounter:
            return 0


    def battleFlow(self,timeCounter,oneshotFlags): #試合の一連を処理します ------------------------------------------------------------------------------------------------------------------------------------------

        if self.playOneMatch(timeCounter,oneshotFlags) == 0: #試合が終わったらそれを呼び出し元に通知
            return 0
        


    def close(self):

        if self.cameraMode == 0:
            self.__infinicam.close()










if __name__ == "__main__":

    battle = Battle()
    deltaTime = 0.0
    counter = 0.0#経過時間のカウンター (試合開始時にリセット)
    oneshotFlags = [True] * 5 #一度だけ実行用のフラグ

    while True:
        start_time = time.time()
        key = cv2.waitKey(1)

        #試合を行う
        ret = battle.battleFlow(counter,oneshotFlags)
        if ret == 0:#試合終了でカウンターをリセットし、再戦
            oneshotFlags = [True] * 5
            counter = 0


        if key == 27:
           break

        #タイマー
        end_time = time.time()
        deltaTime = end_time - start_time
        counter += deltaTime

    battle.close()