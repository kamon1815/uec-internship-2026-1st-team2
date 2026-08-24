import numpy as np
import math
import cv2

import FingerTracking as fin

#  rsp_rate, gesture = jugde.finPos2rpsRate(finger)
#  rsp_rateの戻り値は辞書型です。数値はそれぞれ0~1です。ex) {'rock': 0.5, 'paper': 0.5, 'scissor': 0.5}
#  座標が取得できない場合は{'rock': None, 'paper': None, 'scissor': None}を返します。
#  jugde.showRpsRate(rsp_rate)で各確率のguiを表示します。
# gestureはAIがグーチョキパーを認識した場合、rock, scissor, paperを返します。認識しなかった場合にはNoneを返します。

class JugdeHand():
    def __tangent_angle(self, u: np.ndarray, v: np.ndarray):
        #返り値は0~math.pi
        i = np.inner(u, v)
        n = np.linalg.norm(u) * np.linalg.norm(v)
        c = i / n
        return np.arccos(np.clip(c, -1.0, 1.0))

    def __four_vector2curve(self, p, max_open=math.pi, min_close=0.5*math.pi):
        #p：2次元ベクトルが4つ(掌のベクトル、各関節ごとに指先へ)
        #max_open, min_close: curveRateが1と0にする角度(関節間角度3つの配列を返す)
        #openが1, closeが0
        curveRate=[]
        
        for i in range(3):
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
                        hand_typeRate += abs(curve[i][j]-curveRates[i][j])* importance[type][i][j]/ importance_sum
            rpsRate[type] = 1 - hand_typeRate
        return rpsRate

    def finPos2rpsRate(self, data):
        gesture_text = data.getGesture()
        gesture = None
        if gesture_text == "Open_Palm":
            gesture = 'paper'
        elif gesture_text == "Victory":
            gesture ='scissor'
        elif gesture_text == "Closed_Fist":
            gesture = 'rock'
        if data.getNormalizedPosition().get(0) is not None:
            importance = {'rock': [[0.2,0,0],[0.5,1,1],[0.5,1,1],[0.5,1,1],[0.5,1,1]], 'paper': [[0.5,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]], 'scissor':[[0.5,0,0],[1,1,1],[1,1,1],[0.3,0.5,0.3],[0.3,0.5,0.3]]}
            rps = {'rock': [[0.7,1,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0]], 'paper': [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]], 'scissor':[[0.5,0,0],[1,1,1],[0,0,0],[0,0,0],[0,0,0]]}
            # return self.__judge_rpsRate(self._cal_curveRates(data), importance, rps), gesture
            return self.__judge_rpsRate(self._cal_curveRates(data)), gesture
        else:
            return  {'rock': None, 'paper': None, 'scissor': None}, gesture

    def showRpsRate(self, rpsRate):
        img = np.zeros((300,500,3), dtype='uint8')
        img[:] = (255,255,255)
        if rpsRate['rock'] is not None: 
            img[10:90 ,0:int(rpsRate['rock']*500)] = (0,0,255)
        if rpsRate['rock'] is not None: 
            img[110:190 ,0:int(rpsRate['scissor']*500)] = (0,255,0)
        if rpsRate['rock'] is not None: 
            img[210:290 ,0:int(rpsRate['paper']*500)] = (255,0,0)
        cv2.imshow('rpsRate', img)


if __name__ == '__main__':

    finger = fin.Fingertracking(2.0, 1.0)
    cap = cv2.VideoCapture(0)

    jugde = JugdeHand()

    while True:
        ret,img = cap.read()
        finger.doTracking(img,debug=True,useColor=True)

        rsp_rate, gesture = jugde.finPos2rpsRate(finger)
        jugde.showRpsRate(rsp_rate)
        print(gesture)

        if cv2.waitKey(1) == 27:
            break