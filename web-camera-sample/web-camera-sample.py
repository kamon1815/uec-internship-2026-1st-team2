import cv2 # pip install opencv-python"

cap = cv2.VideoCapture(0)

# ウェブカメラの画像取得
ret, img = cap.read()

# 画像表示
cv2.imshow('test', img)
cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機

cap.release()
cv2.destroyAllWindows()