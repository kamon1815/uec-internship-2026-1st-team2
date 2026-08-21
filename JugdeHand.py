import numpy as np
import math
import cv2

import FingerTracking as fin


def tangent_angle(u: np.ndarray, v: np.ndarray):
    #返り値は0~math.pi
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return np.arccos(np.clip(c, -1.0, 1.0))

def four_vector2curve(p, max_open=3*math.pi, min_close=1.5*math.pi):
    #p：2次元ベクトルが4つ(掌のベクトル、各関節ごとに指先へ)
    #max_open, min_close: curveRateが1と0にする角度(関節間角度3つの合計)
    #openが1, closeが0
    rad=0
    
    for i in range(3):
        if p[i] is not None and p[i+1] is not None:
            rad += tangent_angle(-np.array(p[i]),p[i+1])
        else: #位置が取れないときはmaxとminの中間の値だと考える
            rad += (min_close + (max_open-min_close)/2)/3

    curveRate= (rad-min_close) / (max_open-min_close)
    return max(0, min(curveRate, 1)) #0~1を超えないように

def five_point2four_vector(p):
    #p:(x,y)が5つ、最初が根元、最後が指先
    q=[]
    for i in range(4):
        if p[i] is not None and p[i+1] is not None:
            q.append(np.array(p[i+1]) - np.array(p[i]))
        else:
            q.append(None)
    return q

def cal_curveRates(data):
    curveRates=[]
    fingers_posNum = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for i in range(5):
        p=[]
        for j in range(5):
            p.append(data.getNormalizedPosition().get(fingers_posNum[i][j]))
        curveRate = four_vector2curve(five_point2four_vector(p))
        curveRates.append(curveRate)
    return curveRates

def judge_rpsRate(curveRates):
    rps = {'rock': [0,0,0,0,0], 'paper': [1,1,1,1,1], 'scissor':[0,1,1,0,0]}
    rpsRate={}
    for type, curve in rps.items():
        hand_typeRate=0
        for i in range(5):
            hand_typeRate += abs(curve[i]-curveRates[i])/5
        rpsRate[type] = 1 - hand_typeRate
    return rpsRate

def finPos2rpsRate(data):
    return judge_rpsRate(cal_curveRates(data))

def showRpsRate(rpsRate):
    img = np.zeros((300,500,3), dtype='uint8')
    img[:] = (255,255,255)
    img[10:90 ,0:int(rpsRate['rock']*500)] = (0,0,255)
    img[110:190 ,0:int(rpsRate['scissor']*500)] = (0,255,0)
    img[210:290 ,0:int(rpsRate['paper']*500)] = (255,0,0)
    cv2.imshow('rpsRate', img)


if __name__ == '__main__':

    finger = fin.Fingertracking()
    cap = cv2.VideoCapture(0)

    while True:
        ret,img = cap.read()
        finger.doTracking(img,True)
        showRpsRate(finPos2rpsRate(finger))

        if cv2.waitKey(1) == 27:
            break