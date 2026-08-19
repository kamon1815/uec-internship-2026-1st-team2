import cv2 # need to import extra module "pip install opencv-python"
from pathlib import Path

# 画像読み込み
BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "img/infinicam.png"
img = cv2.imread(path)

# # テキスト描画
# cv2.putText(img,
#             "Hello Python",
#             org=(200, 50),
#             fontFace=cv2.FONT_HERSHEY_DUPLEX,
#             fontScale=1.5,
#             color=(0, 255, 0),
#             thickness=2,
#             lineType=cv2.LINE_AA)

# # リサイズ
# img = cv2.resize(img, dsize=(50, 50))

# 画像表示
cv2.imshow('test', img)

# 画像保存
# cv2.imwrite('sample.png', img)

cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
cv2.destroyAllWindows()