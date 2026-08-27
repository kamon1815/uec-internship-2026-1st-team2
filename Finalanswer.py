import cv2 # need to import extra module "pip install opencv-python"
import pypuclib
from pypuclib import CameraFactory, Camera, XferData, Decoder
from pypuclib import Resolution, PUCException, GPUSetup
import numpy as np
import math
from collections import Counter
import random

class finalanswer():
    #データの受け皿を作成
    def __init__(self,history_len, rock_th=0.77, scissor_th=0.70, paper_th=0.85, Confidencethreshold=0.60, truncation_inc=5.2, rateofchange_inc=3.8, rateofchange_th=0.08, appearance_inc=4.2, appearance_th=0.58):
        self.rate_history =  []
        #履歴リストの最大容量（上限）
        self.history_limit = history_len
        self.current_rates = {'rock': 0.0, 'scissor': 0.0, 'paper': 0.0}
        self.last_rates ={'rock': 0.0, 'scissor': 0.0, 'paper': 0.0} 
        #それぞれのしきい値の設定
        self.thresholds = {
            'rock': rock_th,
            'scissor': scissor_th,
            'paper': paper_th
        }
        #自信率の足切りライン
        self.Confidencethreshold = Confidencethreshold
        #シグモイド関数用のそれぞれの傾き(_inc)と基準値(_th)
        self.truncation_inclination = truncation_inc
        self.ratechange_inclination = rateofchange_inc
        self.b_rateofchange = rateofchange_th
        self.appearance_inclination = appearance_inc
        self.b_appearance = appearance_th
        print("history_limit =", self.history_limit)
        print("rock_th =", self.thresholds["rock"])
        print("scissor_th =", self.thresholds["scissor"])
        print("paper_th =", self.thresholds["paper"])
        print("Confidencethreshold =", self.Confidencethreshold)
        print("truncation_inc =", self.truncation_inclination)
        print("ratechange_inc =", self.ratechange_inclination)
        print("ratechange_th =", self.b_rateofchange)
        print("appearance_inc =", self.appearance_inclination)
        print("appearance_th =", self.b_appearance)
    #初期設定とデータの読み取り
    def  settings_reading(self, rps_rate):
        #過去データの退避
        self.last_rates = self.current_rates.copy()
        #None→0.0にする
        self.current_rates['rock'] = rps_rate.get('rock') if rps_rate.get('rock') is not None else 0.0
        self.current_rates['paper'] = rps_rate.get('paper') if rps_rate.get('paper') is not None else 0.0
        self.current_rates['scissor'] = rps_rate.get('scissor') if rps_rate.get('scissor') is not None else 0.0
        self.rate_history.append(self.current_rates.copy())
        if len(self.rate_history) > self.history_limit:
            self.rate_history.pop(0)  
    #昔のデータをリセット
    def reset_history(self):
        self.rate_history = []
        self.current_rates = {'rock': 0.0,'scissor': 0.0,'paper': 0.0}
        self.last_rates = {'rock': 0.0,'scissor': 0.0,'paper': 0.0}
    

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
         #自信率比較を行うためのシグモイド関数の計算
            if best_canndidate[1] <= 0.0:
                    truncation_sigmoidrate = 0.0
                    return best_canndidate[0], truncation_sigmoidrate
            else:
                if best_canndidate[0] == 'rock':
                   truncation_sigmoidrate = 1 / (1 + math.exp(-self.truncation_inclination * (best_canndidate[1] -self.thresholds.get("rock") )))
                if best_canndidate[0] == 'paper':
                    truncation_sigmoidrate = 1 / (1 + math.exp(-self.truncation_inclination * (best_canndidate[1] -self.thresholds.get("paper") )))
                if best_canndidate[0] == 'scissor': 
                    truncation_sigmoidrate = 1 / (1 + math.exp(-self.truncation_inclination * (best_canndidate[1] -self.thresholds.get("scissor") )))
                return  best_canndidate[0], truncation_sigmoidrate
                                
            
       
    #増減率比較
    def Rateofchange(self):
        #手の候補のストック
        Rateofchange_candidate = []
         #手の形の増減率それぞれ順番にチェック
        for hand in ["rock", "scissor", "paper"]:
            #最大9回の増減率を入れる箱を作成
            change_valallbox = []
            #i-1回目の値-i回目の値を計算し作成した箱に入れる
            for i in range(len(self.rate_history) - 1):    
                change_val = self.rate_history[i + 1][hand] - self.rate_history[i][hand]
                change_valallbox.append(change_val)
            #正の値しか入れない箱を作成    
            plus_only_box = []
            #正の値を作成した箱に入れる
            for val in change_valallbox:
                if val > 0.0:
                    plus_only_box.append(val)
            #正の値のみを計算に入れて平均をとった増減率を作成
            change_val = 0.0
            if len(plus_only_box) > 0:
                change_val = sum(plus_only_box) / len(plus_only_box)
            #変化率が正の値の場合
            if change_val >= 0.0:
                Rateofchange_candidate.append((hand,change_val))
        if len(Rateofchange_candidate)  == 0:
            return None, 0.0
        else:
         #合格したものから順位ずけをさらに行って候補を一つに絞る
            best_canndidate = max(Rateofchange_candidate, key=lambda x: x[1] )
         #自信率比較を行うためのシグモイド関数の計算
            if best_canndidate[1] <= 0.0:
                ratechange_sigmoidrate = 0.0
                return best_canndidate[0], ratechange_sigmoidrate
            else:
                ratechange_sigmoidrate = 1 / (1 + math.exp(-self.ratechange_inclination * (best_canndidate[1] - self.b_rateofchange)))
                return  best_canndidate[0], ratechange_sigmoidrate
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
        # 履歴を逆順（最新が先頭）にして集計する
        counts = Counter(reversed(past_hands))  
            
        #集計して上位1件を取り出し[0]、その外枠を剥ぎ取って、手(文字)と回数(数字)に仕分け
        most_common_hand, most_common_count = counts.most_common(1)[0]
        #もっとも映っている結果はなにかそもそもあるのか
        if most_common_hand is None:
            return None, 0.0
        appearance_rate = most_common_count / len(self.rate_history)
        #自信率比較を行うためのシグモイド関数の計算
        if appearance_rate <= 0.0:
            appearance_sigmoidrate = 0.0
            return most_common_hand, appearance_sigmoidrate
        else:
           appearance_sigmoidrate = 1 / (1 + math.exp(-self.appearance_inclination * (appearance_rate - self.b_appearance)))
           return  most_common_hand, appearance_sigmoidrate




    #最終決定判断
    def get_finalanswer(self):
        # 3つの関数を呼び出す
        hand1, conf1 = self.Truncation()
        hand2, conf2 = self.Rateofchange()
        hand3, conf3 = self.statisticalcomparison()
        print("Truncation   :", hand1, conf1)
        print("Rateofchange :", hand2, conf2)
        print("Statistical  :", hand3, conf3)

        #多数決判定
        votes = []
        if hand1 is not None: votes.append(hand1)
        if hand2 is not None: votes.append(hand2)
        if hand3 is not None: votes.append(hand3)
        print("hand1 =", hand1, "conf1 =", conf1)
        print("hand2 =", hand2, "conf2 =", conf2)
        print("hand3 =", hand3, "conf3 =", conf3)
        print("votes =", votes)
        if len(votes) > 0:
            votes_counts = Counter(votes)
            Majorityrule_hand, Majorityrule_count = votes_counts.most_common(1)[0]
            print("votes_counts =", votes_counts)
            print("Majorityrule_hand =", Majorityrule_hand)
            print("Majorityrule_count =", Majorityrule_count)

            if Majorityrule_count >= 2:
                print(">>> 多数決判定")
                print(">>> result =", Majorityrule_hand)
                return Majorityrule_hand,"多数決判定"
        #多数決ができない場合のシグモイド関数を用いた自信率比較
            elif   Majorityrule_count <= 1: 
                print(">>> 多数決不成立")
                print(">>> 自信率比較へ移行")
                confidencerate = []
                if hand1 is not None: confidencerate.append((hand1, conf1))
                if hand2 is not None: confidencerate.append((hand2, conf2))
                if hand3 is not None: confidencerate.append((hand3, conf3))
                print("confidencerate =", confidencerate)
                if len(confidencerate) > 0:
                    bestconfidencerate_canndidate = max(confidencerate, key=lambda x: x[1] )
                    print("best candidate =", bestconfidencerate_canndidate)
                    print("best hand =", bestconfidencerate_canndidate[0])
                    print("best confidence =", bestconfidencerate_canndidate[1])
                    print("threshold =", self.Confidencethreshold)
                    print("confidence > threshold ?",bestconfidencerate_canndidate[1],">",self.Confidencethreshold,"=",bestconfidencerate_canndidate[1] > self.Confidencethreshold)
                    
                    if bestconfidencerate_canndidate[1] > self.Confidencethreshold:

                        print(">>> 自信率判定を採用")
                        print(">>> result =", bestconfidencerate_canndidate[0])
                        return bestconfidencerate_canndidate[0], "自信率判定"
        #多数決と自信率比較ともにできなかった場合の判定(最後の画像で判断→ランダム)         
                    else:
                        print(">>> 自信率がthreshold未満")
                        if conf1 > 0.5:
                           print(">>> 最後の画像から採用:", hand1)
                           return hand1, "最後の画像から判定"
                        else:
                            print(">>> 最後の画像でも不可 → ランダム")
                            random_gesture = ["rock", "paper", "scissor"]
                            chosenhand = random.choice(random_gesture)
                            return chosenhand,"ランダム"            

                else:
                    if conf1 > 0.5:
                        print(">>> 最後の画像から採用:", hand1)
                        return hand1, "最後の画像判定" 
                    else:
                        print(">>> 最後の画像判定でも不可 → ランダム")
                        random_gesture = ["rock", "paper", "scissor"]
                        chosenhand = random.choice(random_gesture)
                        return chosenhand,"ランダム"            
                                
        else:
            print(">>> 自信率がthreshold未満")
            if conf1 > 0.5:
                print(">>> 最後の画像から採用:", hand1)
                return hand1, "最後の画像から判定"
            else:
                print(">>> 最後の画像でも不可 → ランダム")
                random_gesture = ["rock", "paper", "scissor"]
                chosenhand = random.choice(random_gesture)
                return chosenhand,"ランダム"            
                         
                
          #最終決定判断改良

        







