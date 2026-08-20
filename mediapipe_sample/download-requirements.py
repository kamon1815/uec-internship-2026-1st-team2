# モデルをダウンロード
import urllib.request

url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"

urllib.request.urlretrieve(url, "gesture_recognizer.task")

print("downloaded")