from FingerTracking import Fingertracking
from InfinicamManeger import InfinicamManager
from JugdeHand import JugdeHand
import cv2
import time

class Battle:

    def __init__(self):
        #self.__infinicam = InfinicamManager()
        #self.__infinicam.connect(500,1246,1024,2.0,1)

        ##webカメラでのデバッグ
        self.__cap = cv2.VideoCapture(0)

        self.__tracker = Fingertracking()

        self.__judge = JugdeHand()



    def computeRPS(self, maxtrial = 10,amount = 2, duration = 10 ):

        rsp_rates = []
        rsp_gestures = []
        for i in range(amount):
            for trial in range(maxtrial): #2回目の画像取得。失敗ならmaxtrial回まで再試行。
                #img,defimg = self.__infinicam.get_frame()
                ret,img = self.__cap.read()
                #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                self.__tracker.doTracking(img,debug=True,useColor=False)
                if self.__tracker.getNormalizedPosition() != None:
                    break
            #1回目の画像について判定する
            rsp_rates.append(self.__judge.finPos2rpsRate(self.__tracker))
            #self.__judge.showRpsRate(rsp_rates[i])
            time.sleep(duration /1000)
        return rsp_rates,rsp_gestures


    def playOneMatch(self,dt):
            
            self.__tracker.drawLandmarks()
            print("最初は")
            time.sleep(1)
            print("グー")
            time.sleep(1)
            print("じゃん")
            time.sleep(0.5)
            print("けん")
            time.sleep(0.5)
            print("ぽん")
            rpsrates,gestures = self.computeRPS()

            print(rpsrates)
            time.sleep(5)





    def battleFlow(self):

        img,defimg = self.__infinicam.get_frame()
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        self.__tracker.doTracking(img,debug=True,useColor=False)

        rsp_rate = self.__judge.finPos2rpsRate(self.__tracker)
        self.__judge.showRpsRate(self.__judge.finPos2rpsRate(self.__tracker))

    def close(self):
        self.__infinicam.close()

if __name__ == "__main__":
    battle = Battle()
    deltaTime = 0.0
    while True:
        start_time = time.time()

        key = cv2.waitKey(1)
        #battle.battleFlow()
        battle.playOneMatch(deltaTime)

        if key == 27:
           break
        end_time = time.time()
        deltaTime = end_time - start_time
    battle.close()