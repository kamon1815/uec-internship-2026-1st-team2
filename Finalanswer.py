import cv2 # need to import extra module "pip install opencv-python"
import pypuclib
from pypuclib import CameraFactory, Camera, XferData, Decoder
from pypuclib import Resolution, PUCException, GPUSetup
import numpy as np
import math
from collections import Counter

class finalanswer():
    #データの受け皿を作成
    def __init__(self,max_hold,history_len, rock_th, scissor_th, paper_th):
        self.rate_history =  []
        self.history_limit = history_len
        self.current_rates = {'rock': 0.0, 'scissor': 0.0, 'paper': 0.0}
        self.last_rates ={'rock': 0.0, 'scissor': 0.0, 'paper': 0.0} 
        self.hold_counter = 0
        #それぞれのしきい値の設定
        self.thresholds = {
            'rock': rock_th,
            'scissor': scissor_th,
            'paper': paper_th
        }

    #初期設定とデータの読み取り
    def  settings_reading(self, rps_rate, gesture):
        #過去データの退避
        self.last_rates = self.current_rates.copy()
        #None→0.0にする
        self.current_rates['rock'] = rps_rate.get('rock') if rps_rate.get('rock') is not None else 0.0
        self.current_rates['paper'] = rps_rate.get('paper') if rps_rate.get('paper') is not None else 0.0
        self.current_rates['scissor'] = rps_rate.get('scissor') if rps_rate.get('scissor') is not None else 0.0
        self.rate_history.append(self.current_rates.copy())

        if len(self.rate_history) > self.history_limit:
            self.rate_history.pop(0)
#それぞれの判定
    #数値切り判定
    def  Truncation(self): 
        #手の候補のストック
        Truncation_candidate = []
        #手の形をそれぞれ順番にチェック
        for hand in ["rock", "scissor", "paper"]:
            current_val = self.current_rates[hand]
            hand_threshold = self.thresholds[hand]
        #確率≧しきい値の場合
            if current_val >= hand_threshold:
               Truncation_candidate.append((hand, current_val)) 
        #合格者不在の場合
        if len(Truncation_candidate) == 0:
            return None, 0.0
        else:
        #合格したものから順位ずけをさらに行って候補を一つに絞る
            best_canndidate = max(Truncation_candidate, key=lambda x: x[1] )
        return best_canndidate[0], best_canndidate[1]
    #増減率比較
    def Rateofchange(self):
        #手の候補のストック
        Rateofchange_candidate = []
         #手の形の増減を率それぞれ順番にチェック
        for hand in ["rock", "scissor", "paper"]:
            change_val = self.current_rates[hand] - self.last_rates[hand]
            #変化率が正の値の場合
            if change_val >= 0.0:
                Rateofchange_candidate.append((hand,change_val))
        if len(Rateofchange_candidate)  == 0:
            return None, 0.0
        else:
         #合格したものから順位ずけをさらに行って候補を一つに絞る
            best_canndidate = max(Rateofchange_candidate, key=lambda x: x[1] )
        return best_canndidate[0], best_canndidate[1]
    #統計比較
    def statisticalcomparison(self):
        if len(self.rate_history) == 0:
            return None , 0.0
        #一フレームにおいてどの手が一番出ているかの判断
        past_hands = []
        for frame in self.rate_history:
            top_hand = max(frame, key=frame.get)
            #手が判断ができてるかどうか
            if frame[top_hand] == 0.0: 
                past_hands.append(None)
            else:
                past_hands.append(top_hand)
        counts = Counter(past_hands)  
            
        #集計して上位1件を取り出し[0]、その外枠を剥ぎ取って、手(文字)と回数(数字)に仕分け
        most_common_hand, most_common_count = counts.most_common(1)[0]
        #もっとも映っている結果はなにかそもそもあるのか
        if most_common_hand is None:
            return None, 0.0
        appearance_rate = most_common_count / self.history_limit
        return most_common_hand, appearance_rate
    def get_finalanswer(self, gesture):
        # 3つの関数を呼び出す
        hand1, conf1 = self.Truncation()
        hand2, conf2 = self.Rateofchange()
        hand3, conf3 = self.statisticalcomparison()
        #多数決判定
        votes = []
        if hand1 is not None: votes.append(hand1)
        if hand2 is not None: votes.append(hand2)
        if hand3 is not None: votes.append(hand3)
        
        if len(votes) > 0:
            votes_counts = Counter(votes)
            Majorityrule_hand, Majorityrule_count = votes_counts.most_common(1)[0]
            if Majorityrule_count >= 2:
                return Majorityrule_hand
        #多数決ができない場合の自信率比較
            elif   Majorityrule_count <= 1: 
                confidencerate = []
                if hand1 is not None: confidencerate.append((hand1, conf1))
                if hand2 is not None: confidencerate.append((hand2, conf2))
                if hand3 is not None: confidencerate.append((hand3, conf3))

                if len(confidencerate) > 0:
                    best_canndidate = max(confidencerate, key=lambda x: x[1] )
                    return best_canndidate[0]              











         







