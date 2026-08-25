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
    oneshotFlags = [True] * 6   
    rsp_gestures = []
    rsp_rates = []

    def __init__(self,cameramode = 0, capturemode = 0):
        self.cameraMode = cameramode# 0: inficicamを使用する 1: webカメラを使用する
        self.captureMode = capturemode# 0: 2つの画像で判定(確率の増分での判断に使用)  1: 10枚の画像で判定(統計的判断に使用)

        if self.cameraMode == 0:
            self.__infinicam = InfinicamManager()
            self.__infinicam.connect(988,640,525,2.0,1) #fps , 解像度縦横, コントラスト, 明るさ

        if self.cameraMode == 1:
            self.__cap = cv2.VideoCapture(0)
            self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 525)
            print(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.__tracker = Fingertracking()

        self.__judge = JugdeHand()
        self.__finalans = finalanswer(10,0.5,0.5,0.5,0.5)


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
            self.__finalans.settings_reading(rate)
            if i < amount-1:
                time.sleep(duration /1000)

        return rsp_rates,rsp_gestures




    def computeRPS_Lite(self): # 画像取得からパーセンテージの計算までを行います   maxtrialは撮影失敗時に取り直す回数、 amountは最終的に出力する枚数、durationは撮影間隔


        img = self.getCameraImage()

        self.__tracker.doTracking(img,debug=True,useColor=False)
                
        if self.__tracker.getNormalizedPosition().get(0) == None: #トラッキング成功時はそれを判定へ回す
            return False


        rate,gesture = self.__judge.finPos2rpsRate(self.__tracker)
        self.rsp_rates.append(rate)
        self.rsp_gestures.append(gesture)
        self.__finalans.settings_reading(rate)
        return True



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

    
    def showResultImg(self,ret):
        match ret:
            case "rock":
                print("相手：パー")
            case "scissor":
                print("相手：グー")
            case "paper":
                print("相手：チョキ")     


    
    shotCounter = 0#撮影枚数カウンタ
    shotCounter_timeout = 0#撮影失敗枚数
    prev_shot_delta_time = 0#前回撮影時のcountを記録

    

    def playOneMatch(self,timeCounter):# 1試合分の一連の処理を行います ----------------------------------------------------------------------------------------------------------------------------------

        duration = 10 #撮影間隔
        amount = 7 #撮影枚数
        timeout = 10

        ##カメラからの映像を常に表示
        cv2.imshow(self.battle_window,self.getCameraImage())

        if 0 < timeCounter <= 1:
            if(self.oneshotFlags[0]):
                print("最初は")
                self.oneshotFlags[0] = False
        if 1 < timeCounter <= 2:
            if(self.oneshotFlags[1]):
                print("グー")
                self.oneshotFlags[1] = False

        if 2 < timeCounter <= 2.5:
            if(self.oneshotFlags[2]):
                print("じゃん")
                self.oneshotFlags[2] = False
        if 2.5 <= timeCounter <= 2.9:
            if(self.oneshotFlags[3]):
                print("けん")
                self.oneshotFlags[3] = False
        if 2.85 <= timeCounter:
            start_time = 0
            if self.shotCounter < amount:
                start_time = time.time()
                if self.prev_shot_delta_time == 0:
                    if self.computeRPS_Lite():
                        self.shotCounter += 1

                        self.prev_shot_delta_time = counter
                    elif counter - self.prev_shot_delta_time > timeout / 1000:
                        self.shotCounter += 1
                        self.shotCounter_timeout += 1


                elif counter - self.prev_shot_delta_time  > duration /1000 :
                    self.prev_shot_delta_time = 0


            if self.shotCounter >= amount:#骨格推定と判定
                if self.oneshotFlags[5]:
                    end_time = time.time()
                    print(self.rsp_rates)
                    print(self.rsp_gestures)
                    print("処理時間" + str(end_time - start_time))
                    print("撮影成功枚数" + str(self.shotCounter - self.shotCounter_timeout))
                    #デバッグ用改造済みshowRpsRate
                    #for i in range(len(self.rsp_rates)):
                    #    self.showRpsRate(self.rsp_rates[i],str(i))
                    
                    #print(self.__finalans.get_finalanswer(self.rsp_gestures,None))

                    self.showResultImg(self.__finalans.get_finalanswer(self.rsp_gestures,None))

                    self.oneshotFlags[5] = False
                    
        if 3.0 <= timeCounter:
            if(self.oneshotFlags[4]):
                print("ぽん")
                self.oneshotFlags[4] = False
   
        if 7 < timeCounter:
            return 0


    def battleFlow(self,timeCounter): #試合の一連を処理します ------------------------------------------------------------------------------------------------------------------------------------------

        if self.playOneMatch(timeCounter) == 0: #試合が終わったらそれを呼び出し元に通知
            return 0
        
    def reset(self):
        self.oneshotFlags = [True] * len(self.oneshotFlags) 
        self.prev_shot_delta_time = 0
        self.shotCounter = 0
        self.shotCounter_timeout = 0
        self.rsp_rates = []
        self.rsp_gestures = []

    def close(self):

        if self.cameraMode == 0:
            self.__infinicam.close()










if __name__ == "__main__":

    battle = Battle(0,0)
    deltaTime = 0.0
    counter = 0.0#経過時間のカウンター (試合開始時にリセット)
     #一度だけ実行用のフラグ

    

    while True:
        start_time = time.time()
        key = cv2.waitKey(1)

        #試合を行う
        ret = battle.battleFlow(counter)
        if ret == 0:#試合終了でカウンターをリセットし、再戦
            battle.reset()
            counter = 0


        if key == 27:
           break

        #タイマー
        end_time = time.time()
        deltaTime = end_time - start_time
        counter += deltaTime

    battle.close()