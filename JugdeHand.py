import numpy as np
import math
import cv2
from pathlib import Path
import json

import FingerTracking as fin

#  rsp_rate, gesture = jugde.finPos2rpsRate(finger, camera = 'web')
#  camera='web'はなくても動きます。infinicamでデータ取得した際のパラメータを使いますが、大した差はありません。
#  rsp_rateの戻り値は辞書型です。数値はそれぞれ0~1です。ex) {'rock': 0.5, 'paper': 0.5, 'scissor': 0.5}
#  座標が取得できない場合は{'rock': None, 'paper': None, 'scissor': None}を返します。
#  jugde.showRpsRate(rsp_rate)で各確率のguiを表示します。
# gestureはAIがグーチョキパーを認識した場合、rock, scissor, paperを返します。認識しなかった場合にはNoneを返します。

class JugdeHand():
    def __init__(self):
            BASE_DIR = Path(__file__).resolve().parent
            json_path = BASE_DIR / 'data\\param.json'
            with open(json_path, "r", encoding="utf-8") as f:
                param = json.load(f)
            self.__importance_web = param.get("web").get("importance")
            self.__rps_web = param.get("web").get("rps")
            self.__importance_inf = param.get("infinicam").get("importance")
            self.__rps_inf = param.get("infinicam").get("rps")
    def __tangent_angle(self, u: np.ndarray, v: np.ndarray):
        #返り値は0~math.pi
        i = np.inner(u, v)
        n = np.linalg.norm(u) * np.linalg.norm(v)
        c = i / n
        return np.arccos(np.clip(c, -1.0, 1.0))

    def __four_vector2curve(self, p):
        #p：2次元ベクトルが4つ(掌のベクトル、各関節ごとに指先へ)
        #max_open, min_close: curveRateが1と0にする角度(関節間角度3つの配列を返す)
        #openが1, closeが0
        curveRate=[]
        
        for i in range(3):
            if i == 1:
                min_close = 0.1 * math.pi
            elif i ==2:
                min_close = 0.85 * math.pi
            else:
                min_close = 0.7 * math.pi
            # min_close = 0.5 * math.pi
            max_open = math.pi
            rad = self.__tangent_angle(-np.array(p[i]),p[i+1])
            openRate= (rad-min_close) / (max_open-min_close)
            curveRate.append(max(0, min(openRate, 1))) #0~1を超えないように
        return curveRate 

    def __five_point2four_vector(self, p):
        #p:(x,y)が5つ、最初が根元、最後が指先
        q=[]
        for i in range(4):
            q.append(np.array(p[i+1]) - np.array(p[i]))
        return q

    def _cal_curveRates(self, data):
        curveRates=[]
        fingers_posNum = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
        for i in range(5):
            p=[]
            for j in range(5):
                p.append(np.array(data.getNormalizedPosition().get(fingers_posNum[i][j])) * 100)
    
            curveRate = self.__four_vector2curve(self.__five_point2four_vector(p))
            curveRates.append(curveRate)
        return curveRates

    def __judge_rpsRate(self, curveRates, importance = {'rock': [[0,0,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1]], 'paper': [[0,0,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1]], 'scissor':[[0,0,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]}, rps={'rock': [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]], 'paper': [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]], 'scissor':[[0,0,0],[1,1,1],[1,1,1],[0,0,0],[0,0,0]]}):
        rpsRate={}
        for type, curve in rps.items():
            hand_typeRate=0
            importance_sum = sum(sum(row) for row in importance[type])
            for i in range(5):
                    for j in range(3):
                        hand_typeRate += (abs(curve[i][j]-curveRates[i][j]) /max(abs(curve[i][j]-0),abs(1-curve[i][j])))* importance[type][i][j]/ importance_sum
                        # hand_typeRate += (abs(curve[i][j]-curveRates[i][j]))* importance[type][i][j]/ importance_sum
            rpsRate[type] = 1 - hand_typeRate
        return rpsRate

    def finPos2rpsRate(self, data, camera = 'web'):
        if camera == 'infinicam':
            importance, rps = self.__importance_inf, self.__rps_inf
        else:
            importance, rps = self.__importance_web, self.__rps_web
        gesture_text = data.getGesture()
        gesture = None
        if gesture_text == "Open_Palm":
            gesture = 'paper'
        elif gesture_text == "Victory":
            gesture ='scissor'
        elif gesture_text == "Closed_Fist":
            gesture = 'rock'
        if data.getNormalizedPosition().get(0) is not None:

            if importance is not None and rps is not None:
                return self.__judge_rpsRate(self._cal_curveRates(data), importance, rps), gesture
            return self.__judge_rpsRate(self._cal_curveRates(data)), gesture
        else:
            return  {'rock': None, 'paper': None, 'scissor': None}, gesture

    def showRpsRate(self, rpsRate, output_img = False):
        img = np.zeros((300,500,3), dtype='uint8')
        img[:] = (255,255,255)
        if rpsRate['rock'] is not None: 
            img[10:90 ,0:int(rpsRate['rock']*500)] = (0,0,255)
        if rpsRate['rock'] is not None: 
            img[110:190 ,0:int(rpsRate['scissor']*500)] = (0,255,0)
        if rpsRate['rock'] is not None: 
            img[210:290 ,0:int(rpsRate['paper']*500)] = (255,0,0)
        cv2.imshow('rpsRate', img)
        if output_img == True:
            return img


if __name__ == '__main__':
    from InfinicamManeger import InfinicamManager

    BASE_DIR = Path(__file__).resolve().parent
    output_img_path = BASE_DIR / 'data\\result.png'

    finger = fin.Fingertracking(2.0, 1.0)
    cap = cv2.VideoCapture(0)

    jugde = JugdeHand()
    result_img ={}
    result_rsp_rate = {'rock': [], 'scissor':[], 'paper': []}

    print("Is the Infinicam connected? Please answer True or False.")
    val = input()
    if val == "True":
        infinicam = InfinicamManager()
        infinicam.connect(500,1246,1024,2.0,1)
    infinicam_on = True
    while True:
        if val == "False" or infinicam_on == False:
            ret,img = cap.read()
            finger.doTracking(img,debug=True,useColor=True)
            rsp_rate, gesture = jugde.finPos2rpsRate(finger)
            camera ='web'
        else:
            img,defimg = infinicam.get_frame()
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            finger.doTracking(img,debug=True)
            camera = 'infinicam'
            rsp_rate, gesture = jugde.finPos2rpsRate(finger)
            jugde.showRpsRate(rsp_rate)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('m'):
            infinicam_on = not infinicam_on

        if gesture is not None:
            result_rsp_rate[gesture].append(rsp_rate)


    all_hand_type = True
    for type, rsp_rates in result_rsp_rate.items():
        rsps = {'rock': [], 'scissor':[], 'paper': []}
        rsp_rate = {}
        for i in range(len(rsp_rates)):
            for rate_type, rate in rsp_rates[i].items():
                rsps[rate_type].append(rate)
        for rate_type, rate in rsps.items():
            if len(rsps[rate_type]) <= 0:
                all_hand_type = False
                break
            rsp_rate[rate_type] = np.average(np.array(rsps[rate_type]))
        if all_hand_type == True:        
            result_img[type] = jugde.showRpsRate(rsp_rate, True)

    if all_hand_type == True:        
        img_rsp =np.concatenate((result_img['rock'], result_img['scissor'], result_img['paper']), axis=1)
        cv2.imshow('rpsRate', img_rsp)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            cv2.imwrite(output_img_path, img_rsp)
            print('saved!')