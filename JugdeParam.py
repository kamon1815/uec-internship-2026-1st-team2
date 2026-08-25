import numpy as np
import math
import cv2
import csv
import itertools
from pathlib import Path
import json

from JugdeHand import JugdeHand
from InfinicamManeger import InfinicamManager
from FingerTracking import Fingertracking

def cal_param(data):
    med = np.median(data)
    std = np.std(data)
    return med, 1 / std

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / 'data\\param.csv'
json_path = BASE_DIR / 'data\\param.json'

print("Is the Infinicam connected? Please answer True or False.")
val = input()
if val == "True":
    infinicam = InfinicamManager()
    infinicam.connect(500,1246,1024,2.0,1)
finger = Fingertracking()
cap = cv2.VideoCapture(0)
jugde = JugdeHand()
infinicam_on = True
while True:
    if val == "False" or infinicam_on == False:
        ret,img = cap.read()
        finger.doTracking(img,debug=True,useColor=True)
        camera ='web'
    else:
        img,defimg = infinicam.get_frame()
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        finger.doTracking(img,debug=True)
        camera = 'infinicam'

    gesture = finger.getGesture()
    if gesture == "Open_Palm":
        gesture = 'paper'
    elif gesture == "Victory":
        gesture ='scissor'
    elif gesture == "Closed_Fist":
        gesture = 'rock'
    else:
        gesture = None

    data = [[camera, gesture]]
    
    if finger.getNormalizedPosition().get(0) is not None:
        curveRates = jugde._cal_curveRates(finger)
        data.extend(curveRates)
        if gesture is not None:
            with open (csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(itertools.chain.from_iterable(data)))

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('m'):
        infinicam_on = not infinicam_on

with open(csv_path, encoding="utf-8") as f:
    reader = csv.reader(f)
    raw_data = [row for row in reader]
    csv_data ={
        "web" : {"rock": [], "scissor": [], "paper" : []},
        "infinicam" : {"rock": [], "scissor": [], "paper" : []}
    }
    for row in raw_data:
        if row[1] != '':
            csv_data[row[0]][row[1]].append(list(row[2:17]))

param ={
        "web" :       {"rps": {"rock": [], "scissor": [], "paper" : []},
                        "importance": {"rock": [], "scissor": [], "paper" : []}},
        "infinicam" : {"rps": {"rock": [], "scissor": [], "paper" : []},
                       "importance": {"rock": [], "scissor": [], "paper" : []}},
}

for camera, data in csv_data.items():
    for hand_type, values in data.items():
        trans_values = np.transpose(np.array(values))
        rps_finger, imp_finger = [],[]
        if len(trans_values) ==15:
            for i in range(15):
                rps_param, importance = cal_param(np.array(trans_values[i],dtype=float))
                rps_finger.append(rps_param)
                imp_finger.append(importance)
                # if hand_type != 'scissor':
                    # print(np.median(np.array(trans_values[i],dtype=float)))
                    # print(np.max(np.array(trans_values[i],dtype=float)),np.min(np.array(trans_values[i],dtype=float)),)
                if i%3 == 2:
                    # print()
                    param[camera]["rps"][hand_type].append(rps_finger)
                    param[camera]["importance"][hand_type].append(imp_finger)
                    rps_finger, imp_finger = [],[]

with open(json_path, 'w') as f:
    json.dump(param, f, indent=2)


