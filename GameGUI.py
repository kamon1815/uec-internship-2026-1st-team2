import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageOps
from pathlib import Path
import time

from Battle import Battle


class GameGUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.setting_state = False

        self.master.title("GameGUI") # ウィンドウのタイトル
        self.master.geometry("900x500") # ウィンドウのサイズ
        self.master.protocol("WM_DELETE_WINDOW", self.__on_closing)

        self.debug = False

        BASE_DIR = Path(__file__).resolve().parent
        image_path = [BASE_DIR / 'image\\rock.png', BASE_DIR / 'image\\scissors.png',BASE_DIR / 'image\\paper.png']
        self.__canvas_img_computer = {'rock' : Image.open(image_path[0]), 'scissor': Image.open(image_path[1]), 'paper': Image.open(image_path[2])}

        self.__img_cam_0, self.__img_cam_1, self.__img_computer_0,self.__img_computer_1 = None, None, None, None
        self.__img_cam_0_state, self.__img_computer_0_state = False, False
        self.__text = None

        self.__canvas_cam = tk.Canvas(self.master, bg= "white", bd=0, highlightthickness = 0)
        self.__canvas_computer = tk.Canvas(self.master, bg= "white", bd=0, highlightthickness = 0)
        self.__canvas_setting("camera",False)
        self.__canvas_setting("computer",False)

        self.__start_button = tk.Button(self.master, text= "start", bg = "yellow", command= self.__start_button_process)
        self.__start_button.place(relx=0.85, rely=0.8, relwidth=0.1, relheight=0.05)
        self.__started = False

        setting_button = tk.Button(self.master, text= "setting", bg="green", command= self.__setting_button_process)
        setting_button.place(relx=0.85, rely=0.9, relwidth=0.1, relheight=0.05)
        

        self.__voice_text = tk.Label(self.master, text=self.__text,bg='white') 
        self.__voice_text.place(relx=0.25, rely=0.7, relwidth=0.5, relheight=0.25)
        self.__voice_text.bind("<Configure>", self.__resize_font)

        self.__loop_val = False
        self.__loop = tk.Button(self.master, text="連続じゃんけん", bg= "gray", command=self.__loop_button_process)
        self.__loop.place(relx=0.85, rely=0.7, relwidth=0.1, relheight=0.05)


        self.__battle = Battle(0,0, self)
        config = {"camera":0,"contrast":1,"brightness":1,"auto":1}
        self.__battle.changeCameraConfig(config)
        self.__deltaTime = 0.0
        self.__counter = 0.0#経過時間のカウンター (試合開始時にリセット)

        self.__battle_update_time = 20

        self.__timer_ID = []

        self.update()


    def setCameraImage(self, img):
        if self.__img_cam_0_state == False:
            self.__img_cam_0 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.__resize_img(img, "camera"), cv2.COLOR_BGR2RGB)))
        else:
            self.__img_cam_1 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.__resize_img(img, "camera"), cv2.COLOR_BGR2RGB)))
        self.__canvas_setting("camera")

    def changeHand(self, hand):
        if hand == None:
            img = None
        else:
            img = ImageTk.PhotoImage(self.__resize_img(self.__canvas_img_computer[hand], "computer"))
        if self.__img_computer_0_state == False:
            self.__img_computer_0 = img
        else:
            self.__img_computer_1 = img
        self.__canvas_setting("computer")

    def changeText(self, text):
        self.__text = text
        self.__voice_text.config(text =self.__text)
 

    def __canvas_setting(self, img_type, reshow= True):
        if img_type =="camera":
            canvas = self.__canvas_cam
            if reshow == False:
                canvas.place(relx=0.05, rely=0.05, relwidth=0.4, relheight=0.6)
        elif img_type =="computer":
            canvas =self.__canvas_computer
            if reshow == False:
                canvas.place(relx=0.55, rely=0.05, relwidth=0.4, relheight=0.6)
                
        # キャンバスのサイズを取得
        canvas.update() # Canvasのサイズを取得するため更新しておく
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # 画像の描画
        if reshow == True:
            canvas.delete(img_type)
        if img_type =="camera":
            if self.__img_cam_0_state == False:
                img = self.__img_cam_0
            else:
                img = self.__img_cam_1
            self.__img_cam_0_state = not self.__img_cam_0_state
        elif img_type =="computer":
            if self.__img_computer_0_state == False:
                img = self.__img_computer_0
            else:
                img = self.__img_computer_1
            self.__img_computer_0_state = not self.__img_computer_0_state
        else:
            img = None
        if img is not None:
            canvas.create_image(
                canvas_width / 2,       # 画像表示位置(Canvasの中心)
                canvas_height / 2,  
                image=img,  # 表示画像データ
                tag= img_type
            )
      

    def __start_button_process(self):
        self.__started = True
        self.__battle.reset()
        self.__counter = 0
        self.__start_button.config(bg="gray")


    def __loop_button_process(self):
        self.__loop_val = not self.__loop_val
        if self.__loop_val:
            self.__loop.config(bg="red")
        else:
            self.__loop.config(bg="gray")
    

    def __setting_button_process(self):
        self.setting_state = True
        self.__battle.close()
        for i in range(len(self.__timer_ID)):
            self.master.after_cancel(self.__timer_ID[i])
        self.__timer_ID.clear()
        self.master.destroy()

    def __resize_img(self, img, type):
        if type == "camera":
            h, w, _ = img.shape
            self.__canvas_cam.update()
            canvas_width = self.__canvas_cam.winfo_width()
            canvas_height = self.__canvas_cam.winfo_height()       

            if w / h > canvas_width / canvas_height:
                img_result = cv2.resize(img, (0, 0), fx=canvas_width/w, fy=canvas_width/w)
            else:
                img_result = cv2.resize(img, (0, 0), fx=canvas_height/h, fy=canvas_height/h)

            self.__canvas_cam.config(width=canvas_width, height=canvas_height)
       

        elif type == "computer":
            self.__canvas_computer.update()
            canvas_width = self.__canvas_computer.winfo_width()
            canvas_height = self.__canvas_computer.winfo_height()
            w, h, = img.width, img.height

            if w / h > canvas_width / canvas_height:
                img_result = ImageOps.scale(img, canvas_width/w)
            else:
                img_result = ImageOps.scale(img, canvas_height/h)
        return img_result

    
    def __resize_font(self, event):
        # ラベルの現在の幅と高さを取得
        self.__voice_text.update()
        w = self.__voice_text.winfo_width()
        h = self.__voice_text.winfo_height()
        
        # サイズが極端に小さい場合は処理しない
        if w < 10 or h < 10:
            return
            
        # 幅と高さの小さい方に合わせてフォントサイズを決める（余白を考慮）
        new_font_size = int(min(w, h) * 0.5)
        if new_font_size < 1:
            new_font_size = 1
            
        # フォントを更新
        self.__voice_text.config(font=("Helvetica", new_font_size))

    def __on_closing(self):
        self.__battle.close()
        for i in range(len(self.__timer_ID)):
            self.master.after_cancel(self.__timer_ID[i])
        self.__timer_ID.clear() 
        self.master.destroy()


    def __battle_update(self):
        start_time = time.time()
        #試合を行う
        
        ret = self.__battle.battleFlow(self.__counter, (not self.__started)) #retが0で試合終了を示す
        if ret == 0:#試合終了でカウンターをリセット
            start_time = 0
            end_time = 0
            self.__deltaTime = 0
            self.__start_button.config(bg="yellow")
            if self.__loop_val:
                self.__battle.reset()
                self.__counter = 0
                self.__start_button.config(bg="gray")
            return

        #タイマー
        end_time = time.time()
        self.__deltaTime = end_time - start_time
        self.__counter += self.__deltaTime + (self.__battle_update_time)/1000

    def update(self):
        self.__battle_update()
        self.__timer_ID.append(self.master.after(self.__battle_update_time, self.update))

    def changeConfig(self, config):
        self.__battle.changeCameraConfig(config)





if __name__ == "__main__":

    # ====
    root = tk.Tk()
    # root.title("GameGUI") # ウィンドウのタイトル
    # root.geometry("900x600")                     # ウィンドウのサイズ
    game = GameGUI(master=root)
    config = {"camera":0,"contrast":1,"brightness":1,"auto":1}
    game.changeConfig(config)
    # メインループの開始
    game.mainloop()
    # ====

    print(game.setting_state) #setting画面に行くかゲームを終了するか

    # if game.setting_state:
    #     root = tk.Tk() #まず土台となるウィンドウ(tk.Tk)を作成し、変数rootに代入
    #     game = GameGUI(master=root)
    #     game.mainloop()

 