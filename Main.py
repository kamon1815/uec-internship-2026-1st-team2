from SettingGUI import SettingGUI
from GameGUI import GameGUI
import tkinter as tk
from enum import Enum
import cv2

class GameState(Enum):
    SETTING = 1
    GAME = 2



def main():
    state = GameState.SETTING
    settingGUI  = SettingGUI()
    config = {"camera":0,"contrast":1,"brightness":1,"auto":1}

    while True:
        if state == GameState.SETTING:
            c,pressed = settingGUI.update()
            if pressed:
                config = c
                settingGUI.close()
                state = GameState.GAME
            key = cv2.waitKey(1)

        if state == GameState.GAME:
            root = tk.Tk()
            # root.title("GameGUI") # ウィンドウのタイトル
            # root.geometry("900x600")                     # ウィンドウのサイズ
            game = GameGUI(master=root)
            
            game.changeConfig(config)
            # メインループの開始
            game.mainloop()

            if game.setting_state:
                settingGUI  = SettingGUI()
                state = GameState.SETTING
            else:
                break

    


if __name__ == "__main__":
    main()