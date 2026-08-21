import mediapipe as mp
import numpy as np
import cv2
from  InfinicamManeger import InfinicamManager

# 使い方の例
# finger = Fingertracking()
# cap = cv2.VideoCapture(0)
# while True:
#    ret,img = cap.read()
#    finger.doTracking(img)
#    print(finger.getRawPosition().get(0) )
#    if cv2.waitKey(1) == 27:
#        break
#
#これで出力に各検出ポイントの座標が表示されます。
#Fingertracking(1.0,1.0)のように左から順にコントラスト、明るさを設定できます。コントラストは1より大きいと高くなります。
#デバッグ用にfinger.doTracking(img,debug = True)にすると、画像に図示されるはずです
#
#finger.doTracking(img)の戻り値は辞書形式です。
#finger.getRawPosition.get(0)で手首の位置の生データが取れます。トラッキング失敗時はKeyerrorになるため、get()で辞書から値をとることをお勧めします。
#finger.getNormalizedPosition().get(0)で手首の位置の正規化済みの位置が取れます。



class Fingertracking():

    __mp_hands = mp.tasks.vision.HandLandmarksConnections
    __mp_drawing = mp.tasks.vision.drawing_utils
    __mp_drawing_styles = mp.tasks.vision.drawing_styles

    __contrast = 1.0
    __light =1.0

    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)


    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # ランドマーク描画
            self.__mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.__mp_hands.HAND_CONNECTIONS,
                self.__mp_drawing_styles.get_default_hand_landmarks_style(),
                self.__mp_drawing_styles.get_default_hand_connections_style()
            )

            # テキスト表示位置
            self.__height, self.__width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            text_x = int(min(x_coordinates) * self.__width)
            text_y = int(min(y_coordinates) * self.__height) - self.MARGIN

            # 元の認識結果
            handedness = handedness_list[idx][0].category_name

            # 左右反転しているので入れ替え
            if handedness == "Left":
                handedness_text = "Right"
            else:
                handedness_text = "Left"

            cv2.putText(
                annotated_image,
                handedness_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                self.FONT_SIZE,
                self.HANDEDNESS_TEXT_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA
            )

        return annotated_image

    def doTracking(self,img,debug = False,colorFillter = cv2.COLORMAP_JET,useColor=False):

        edge = cv2.blur(img,(5,5))
        edge = cv2.Canny(edge,30,80)
        edge = cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)

        mixed = cv2.addWeighted(src1=img, alpha=1.0, src2=edge, beta=0.5, gamma=0)
        frame = cv2.convertScaleAbs(mixed, alpha=self.__contrast, beta=self.__light)
        if useColor == False:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            mappedimg = cv2.applyColorMap(gray, colorFillter)
            frame = cv2.cvtColor(mappedimg, cv2.COLOR_BGR2RGB)
        else: 
            frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # mediapipe.Image に変換
        mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img
           )
        # 手検出
        self.raw_results = self.recognizer.recognize(mp_image)

        # 検出結果描画
        if debug:
            if self.raw_results.hand_landmarks:
                frame = self.draw_landmarks_on_image(frame, self.raw_results)
                #print(results.hand_landmarks)
                # RGB -> BGR
            frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )
            cv2.imshow("Hand Tracking", frame)


    def getRawTrackingData(self):
        return self.raw_results

    # 生の状態で座標を返します。 検出ポイント番号がラベル、xyz座標のリストが要素の辞書が戻り値です。
    def getRawPosition(self):
        pos = {}
    
        if not self.raw_results or not self.raw_results.hand_landmarks:
            return pos 
            
        first_hand_landmarks = self.raw_results.hand_landmarks[0]
        
        for i, dat in enumerate(first_hand_landmarks):
            pos.update({i: (dat.x, dat.y, dat.z)})
            
        return pos

    #Y軸のスケールをX軸、Z軸に揃えたときの座標を返します。
    def getNormalizedPosition(self):
        pos = {}

    
        if not self.raw_results or not self.raw_results.hand_landmarks:
            return pos 
            
        first_hand_landmarks = self.raw_results.hand_landmarks[0]

        ratio = self.__height / self.__width
        
        for i, dat in enumerate(first_hand_landmarks):
            pos.update({i: (dat.x, dat.y * ratio, dat.z)})
            
        return pos


    def __init__(self,contrast = 1.0, light = 1.0):
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = "gesture_recognizer.task"

        self.__contrast = contrast
        self.__light = light

        options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=VisionRunningMode.IMAGE
        )

        self.recognizer = GestureRecognizer.create_from_options(options)


#デバッグ用
if __name__ == "__main__":
    finger = Fingertracking(1.8,1)
    #cap = cv2.VideoCapture(0)
    infinicam = InfinicamManager()
    infinicam.connect(800)
    filter = cv2.COLORMAP_JET
    while True:
        #ret,img = cap.read()
        img = infinicam.get_frame()
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        key = cv2.waitKey(1)
        if key ==  ord('q'):
            filter = cv2.COLORMAP_BONE
        if key ==  ord('w'):
            filter = cv2.COLORMAP_JET
        if key ==  ord('e'):
            filter = cv2.COLORMAP_VIRIDIS
        if key ==  ord('r'):
            filter = cv2.COLORMAP_PINK
        if key ==  ord('t'):
            filter = cv2.COLORMAP_HOT


        finger.doTracking(img,debug = True,colorFillter=filter,useColor=False)
        print(finger.getNormalizedPosition().get(0) )

        if key == 27:
           break

    infinicam.close()





