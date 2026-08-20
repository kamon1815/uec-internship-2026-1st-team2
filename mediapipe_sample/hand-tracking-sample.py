import mediapipe as mp
import numpy as np
import cv2


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # ランドマーク描画
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # テキスト表示位置
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

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
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image

import cv2
import mediapipe as mp

def camera_loop(cap, recognizer):
    while True:
        success, frame = cap.read()

        if not success:
            break

        # 左右反転
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


        # mediapipe.Image に変換
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # 手検出
        results = recognizer.recognize(mp_image)

        # 検出結果描画
        if results.hand_landmarks:
            annotated_rgb = draw_landmarks_on_image(rgb, results)
            print(results.hand_landmarks)
            # RGB -> BGR
            frame = cv2.cvtColor(
                annotated_rgb,
                cv2.COLOR_RGB2BGR
            )

        cv2.imshow("Hand Tracking", frame)

        # ESCで終了
        if cv2.waitKey(1) == 27:
            break

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "gesture_recognizer.task"

options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.IMAGE
)

with GestureRecognizer.create_from_options(options) as recognizer:

    # カメラ起動
    cap = cv2.VideoCapture(0)

    try:
        camera_loop(cap, recognizer)

    finally:
        # エラー時でもちゃんと閉じる
        cap.release()
        cv2.destroyAllWindows()