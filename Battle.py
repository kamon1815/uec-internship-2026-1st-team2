from FingerTracking import Fingertracking
from InfinicamManeger import InfinicamManager
from JugdeHand import JugdeHand
import cv2

class Battle:

    def __init__(self):
        self.__infinicam = InfinicamManager()
        self.__infinicam.connect(500,1246,1024,2.0,1)

        self.__tracker = Fingertracking()

        self.__judge = JugdeHand()

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
    while True:
        key = cv2.waitKey(1)
        battle.battleFlow()


        if key == 27:
           break
    battle.close()