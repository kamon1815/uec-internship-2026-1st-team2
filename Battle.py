#試合関係を処理します
#main内のBattle(1,0) の左側の引数を0にするとinfinicamで動作確認ができます
#main内のBattle(1,0) の左側の引数を1にすると10枚の連番での検証ができます
#playOneMatch内のcomputeRPS(40,2,100)の引数は、撮影失敗時に取り直す回数、最終的に出力する枚数、その撮影間隔となっています。
#この場合は、100ミリ秒間隔で2回の判定を行い、撮影失敗時はそれぞれ最大40回まで再撮影します
#戻り値は   確率(辞書型),ジェスチャー(文字列)の順です。詳しくはJugdeHandをご確認ください。
#
#Battle.pyを実行するとCUIでのじゃんけんの確認ができます。ESCキーで終了します。

from FingerTracking import Fingertracking
from InfinicamManeger import InfinicamManager
from JugdeHand import JugdeHand
from Finalanswer import finalanswer
import numpy as np
import cv2
import time


class Battle:

    battle_window = "BattleWindow"


    def __init__(self,cameramode = 0, capturemode = 0):
        self.cameraMode = cameramode# 0: inficicamを使用する 1: webカメラを使用する
        self.captureMode = capturemode# 0: 2つの画像で判定(確率の増分での判断に使用)  1: 10枚の画像で判定(統計的判断に使用)

        if self.cameraMode == 0:
            self.__infinicam = InfinicamManager()
            self.__infinicam.connect(500,640,525,2.0,1) #fps , 解像度縦横, コントラスト, 明るさ

        if self.cameraMode == 1:
            self.__cap = cv2.VideoCapture(0)
            self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 525)
            print(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.__tracker = Fingertracking()

        self.__judge = JugdeHand()
        self.__finalans = finalanswer(10,0.5,0.5,0.5)


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
                    #print(trial)#試行回数を表示 (デバッグ)
                    break
 
            #i回目の画像について判定する
            rate,gesture = self.__judge.finPos2rpsRate(self.__tracker)
            rsp_rates.append(rate)
            rsp_gestures.append(gesture)
            self.__finalans.settings_reading(rate,gesture)
            if i < amount-1:
                time.sleep(duration /1000)

        return rsp_rates,rsp_gestures



    def computeRPS_Lite(self, counter, maxtrial = 40,amount = 2, duration = 100 ): # 画像取得からパーセンテージの計算までを行います   maxtrialは撮影失敗時に取り直す回数、 amountは最終的に出力する枚数、durationは撮影間隔

        rsp_rates = []
        rsp_gestures = []
        for i in range(amount):
            for trial in range(maxtrial): #i回目の画像取得。失敗ならmaxtrial回まで再試行。

                img = self.getCameraImage()

                self.__tracker.doTracking(img,debug=True,useColor=False)
                
                if self.__tracker.getNormalizedPosition().get(0) != None: #トラッキング成功時はそれを判定へ回す
                    #print(trial)#試行回数を表示 (デバッグ)
                    break
 
            #i回目の画像について判定する
            rate,gesture = self.__judge.finPos2rpsRate(self.__tracker)
            rsp_rates.append(rate)
            rsp_gestures.append(gesture)
            self.__finalans.settings_reading(rate,gesture)
            if i < amount-1:
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

    
    def showResultImg(ret):
        match ret:
            case 0:
                print("グー")
            case 1:
                print("チョキ")
            case 2:
                print("パー")     



    def playOneMatch(self,timeCounter,oneshotFlags):# 1試合分の一連の処理を行います ----------------------------------------------------------------------------------------------------------------------------------

        ##カメラからの映像を常に表示
        cv2.imshow(self.battle_window,self.getCameraImage())

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
        if 2.90 <= timeCounter <= 3.5:

            if(oneshotFlags[4]):
                start_time = time.time()

                print("ぽん")
                #ここで骨格認識と判定
                if self.captureMode == 0:# 2枚での増分による判定
                    rpsrates,gestures = self.computeRPS(maxtrial=10,amount=2,duration=200)

                if self.captureMode == 1:# 10枚での統計的判定
                    rpsrates,gestures = self.computeRPS(maxtrial=5,amount=10,duration=200 / 10)
                end_time = time.time()
                print(rpsrates)
                print(gestures)
                print("処理時間" + str(end_time - start_time))

                print(self.__finalans.get_finalanswer(gestures,None))

                #デバッグ用改造済みshowRpsRate
                for i in range(len(rpsrates)):
                    self.showRpsRate(rpsrates[i],str(i))
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

    battle = Battle(1,1)
    deltaTime = 0.0
    counter = 0.0#経過時間のカウンター (試合開始時にリセット)
    oneshotFlags = [True] * 6 #一度だけ実行用のフラグ

    

    while True:
        start_time = time.time()
        key = cv2.waitKey(1)

        #試合を行う
        ret = battle.battleFlow(counter,oneshotFlags)
        if ret == 0:#試合終了でカウンターをリセットし、再戦
            oneshotFlags = [True] * 6
            counter = 0


        if key == 27:
           break

        #タイマー
        end_time = time.time()
        deltaTime = end_time - start_time
        counter += deltaTime

    battle.close()