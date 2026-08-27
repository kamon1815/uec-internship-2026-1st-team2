#試合関係を処理します
#main内のBattle() の左側の引数を0にするとinfinicamで動作確認ができます
#battle = Battle()でインスタンスを作った後、
#config = {"camera":0,"contrast":1,"brightness":1,"auto":1} camera = 0でinfinicam、1でwebカメラを使用。auto = 1で自動コントラスト調整 0で無効
#battle.changeCameraConfig(config)
#とすることで設定を反映できます。
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
    rsp_gestures = [] #ジェスチャー認識の結果
    rsp_rates = [] # 手の確率の推定結果

    #初期設定
    def __init__(self,cameramode = 0, capturemode = 0, gui = None):
        self.cameraMode = cameramode# 0: inficicamを使用する 1: webカメラを使用する
        self.captureMode = capturemode# 0: 2つの画像で判定(確率の増分での判断に使用)  1: 10枚の画像で判定(統計的判断に使用)

        self.config = {"camera":0,"contrast":1,"brightness":1,"auto":1}

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

        self.__tracker = Fingertracking(1.0,1.0)

        self.__judge = JugdeHand()
        self.__finalans = finalanswer(10)

        self.__gui = gui

    def changeCameraConfig(self,config):#カメラの設定変更
        self.config = config

    def setCameraContrastAndBrightness(self):
        if self.cameraMode == 0:#コントラストや明るさは別クラス担当なのでそっちへ投げる
            self.__infinicam.configurateCameraImage(self.config["contrast"],self.config["brightness"])#カメラの設定変更(仮)
        else:
            self.__tracker.configurateContrastAndBrightness(self.config["contrast"],self.config["brightness"])


    #現在のカメラ設定に応じてカメラから画像を取得する -------------------------------------------------------------------
    def getCameraImage(self):

        if self.cameraMode == 0: # infinicamを使う場合
            if self.__infinicam != None: 
                img,defimg = self.__infinicam.get_frame()
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            else: 
                self.cameraMode = 1 # infinicamが接続されていないならwebカメラを使うようにする

        if self.cameraMode == 1: # webカメラを使う場合
            ret,img = self.__cap.read()

        if self.config.get("auto")== 1:#自動コントラスト調整がオンならその処理をする。
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            min_val, max_val, _, _ = cv2.minMaxLoc(gray)
            contrast = 255 / (max_val - min_val)    
            self.changeCameraConfig({"camera":self.cameraMode,"contrast":contrast, "brightness":self.config["brightness"],"auto":1})
        return img


    #カメラ切り替えの機能 ---------------------------------------------------------------------------------------------
    def changeCamera(self,config):#c = 0でinfinicam、c = 1 でwebカメラ
        self.cameraMode = config.get("camera")



    #骨格推定を実行 (旧版)  ------------------------------------------------------------------------------------------
    def computeRPS(self, maxtrial = 40,amount = 2, duration = 100 ): # 画像取得からパーセンテージの計算までを行います   maxtrialは撮影失敗時に取り直す回数、 amountは最終的に出力する枚数、durationは撮影間隔

        rsp_rates = []
        rsp_gestures = []
        for i in range(amount):
            for trial in range(maxtrial): #i回目の画像取得。失敗ならmaxtrial回まで再試行。

                img = self.getCameraImage()

                if self.__gui is not None and self.__gui.debug == False:
                    self.__tracker.doTracking(img,debug=False,useColor=True)
                else:
                    self.__tracker.doTracking(img,debug=True,useColor=True)
                
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
            if self.__gui is not None:
                self.fingertracker.doTracking(img,debug=False)
            else:
                self.fingertracker.doTracking(img,debug=True)
        return rsp_rates,rsp_gestures



    #骨格推定と確率の評価を実行 (改良版) -------------------------------------------------------------------------------------------------
    def computeRPS_Lite(self): # 画像取得からパーセンテージの計算までを行います


        img = self.getCameraImage()
        #骨格推定を実行
        if self.__gui is not None and self.__gui.debug == False:
            self.__tracker.doTracking(img,debug=False)
        else:
            self.__tracker.doTracking(img,debug=True)
                
        if self.__tracker.getNormalizedPosition().get(0) == None: #トラッキング成功時はそれを判定へ回す
            return False

        #結果を結果判定に登録
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

        if self.__gui is None:
            cv2.imshow(name, img)


    #入力した結果に対して勝つ手を表示する -------------------------------------------------------------------------------------------------------------
    def showResultImg(self,ret):
        match ret:
            case "rock":
                print("相手：パー")     
                if self.__gui is not None:
                    self.__gui.changeHand("paper")
            case "scissor":
                print("相手：グー")
                if self.__gui is not None:
                    self.__gui.changeHand("rock")
            case "paper":
                print("相手：チョキ")  
                if self.__gui is not None:
                    self.__gui.changeHand("scissor") 




    #カウンタやフラグなど。1試合ごとにリセットされる    
    shotCounter = 0#撮影枚数カウンタ
    shotCounter_timeout = 0#撮影失敗枚数
    prev_shot_delta_time = 0#前回撮影時のcountを記録
    oneshotFlags = [True] * 6   #タイムラインで1度だけ実行部分を作るためのフラグ

    # 1試合分の一連の処理を行います ----------------------------------------------------------------------------------------------------------------------------------
    def playOneMatch(self,timeCounter, beforeStart=False):#timeCounterには試合開始時を0とした秒数が入ります。

        duration = 10 #撮影間隔
        amount = 10 #撮影枚数
        timeout = 20#撮影失敗時の再試行を許す最大時間

        ##カメラからの映像を常に表示
        img = self.getCameraImage()

        if self.__gui is not None:
            if 3.3 <= timeCounter < 7 and beforeStart == False: #手を出した瞬間の映像を少し保持しておく
                img = self.__img_pon
            else:
                self.__img_pon = img
            self.__gui.setCameraImage(img)
        else:
            cv2.imshow(self.battle_window,img)
        self.setCameraContrastAndBrightness()

        if beforeStart:
            return
        
        if 0 < timeCounter <= 1:
            if(self.oneshotFlags[0]):#1度だけ実行
                print("最初は")
                if self.__gui is not None:
                    self.__gui.changeText("最初は")
                self.oneshotFlags[0] = False
        if 1 < timeCounter <= 2:
            if(self.oneshotFlags[1]):#1度だけ実行
                print("グー")
                if self.__gui is not None:
                    self.__gui.changeText("グー")
                    self.__gui.changeHand("rock")
                self.oneshotFlags[1] = False

        if 2 < timeCounter <= 2.5:
            if(self.oneshotFlags[2]):#1度だけ実行

                print("じゃん")
                if self.__gui is not None:
                    self.__gui.changeText("じゃん")
                    self.__gui.changeHand(None)
                self.oneshotFlags[2] = False
        if 2.5 <= timeCounter <= 2.9:
            if(self.oneshotFlags[3]):#1度だけ実行
                print("けん")
                if self.__gui is not None:
                    self.__gui.changeText("けん")
                self.oneshotFlags[3] = False

        if 2.68 <= timeCounter:

            start_time = 0#処理時間計測用

            if self.shotCounter < amount: #撮影枚数以下なら撮影を試す
                start_time = time.time()#処理時間計測用

                if self.prev_shot_delta_time == 0: #1枚目を撮影
                    if self.computeRPS_Lite():
                        self.shotCounter += 1 #撮影成功時は撮影枚数カウンタを増やす

                        self.prev_shot_delta_time = timeCounter #次のフレームの撮影まで待機させたいのでカウンタでいろいろやる。

                    elif timeCounter - self.prev_shot_delta_time > timeout / 1000: #撮影失敗時は再試行するが、指定時間を超えたら撮影をあきらめる
                        self.shotCounter += 1
                        self.shotCounter_timeout += 1


                elif timeCounter - self.prev_shot_delta_time  > duration /1000 : #撮影間隔を空けるための部分
                    self.prev_shot_delta_time = 0


            if self.shotCounter >= amount:#骨格推定と判定
                if self.oneshotFlags[5]:#1度だけ実行
                    end_time = time.time()#処理時間計測
                    print(self.rsp_rates)
                    print(self.rsp_gestures)
                    print("処理時間" + str(end_time - start_time))
                    print("撮影成功枚数" + str(self.shotCounter - self.shotCounter_timeout))

                    #デバッグ用改造済みshowRpsRate
                    for i in range(len(self.rsp_rates)):
                        self.showRpsRate(self.rsp_rates[i],str(i))
                    
                    #print(self.__finalans.get_finalanswer(self.rsp_gestures,None))
                    ret_ges,ret_way = self.__finalans.get_finalanswer(self.rsp_gestures,"未検出")
                    self.showResultImg(ret_ges)#最終結果を結果表示部分に渡す
                    print(ret_way)
                    self.oneshotFlags[5] = False
                    
        if 3.0 <= timeCounter:
            if(self.oneshotFlags[4]):
                print("ぽん")
                if self.__gui is not None:
                    self.__gui.changeText("ぽん")
                self.oneshotFlags[4] = False
   
        if 7 < timeCounter:
            if self.__gui is not None:
                self.__gui.changeText("")
                self.__gui.changeHand(None)
            return 0


    #試合の一連を処理します ------------------------------------------------------------------------------------------------------------------------------------------
    def battleFlow(self,timeCounter, beforeStart=False): 


        if self.playOneMatch(timeCounter, beforeStart) == 0: #試合が終わったらそれを呼び出し元に通知
            return 0
        return 1

    #フラグなどをリセットします
    def reset(self):
        self.oneshotFlags = [True] * len(self.oneshotFlags) 
        self.prev_shot_delta_time = 0
        self.shotCounter = 0
        self.shotCounter_timeout = 0
        self.rsp_rates = []
        self.rsp_gestures = []

    #infinicamの終了処理
    def close(self):
        if self.__infinicam != None:
            self.__infinicam.close()










if __name__ == "__main__":

    battle = Battle(0,0)
    config = {"camera":0,"contrast":1,"brightness":1,"auto":1}
    battle.changeCameraConfig(config)
    deltaTime = 0.0
    counter = 0.0#経過時間のカウンター (試合開始時にリセット)


    while True:
        start_time = time.time()
        key = cv2.waitKey(1)

        #試合を行う
        
        ret = battle.battleFlow(counter) #retが0で試合終了を示す
        if ret == 0:#試合終了でカウンターをリセットし、再戦
            battle.reset()
            counter = 0
            start_time = 0
            end_time = 0
            deltaTime = 0
            continue

        #カメラ切り替え
        if key == ord('q'):
            battle.changeCamera(0)
        if key == ord('w'):
            battle.changeCamera(1)

        if key == 27:#ESCキーで終了
           break

        #タイマー
        end_time = time.time()
        deltaTime = end_time - start_time
        counter += deltaTime

    battle.close()