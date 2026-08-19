import time
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# 推論を行うデバイス、モデルのパス設定
BASE_DIR = Path(__file__).resolve().parent
MODE = "npu_fp16"                       # cpu_fp32 / gpu_fp32 / gpu_fp16 / npu_fp16
MODEL = BASE_DIR / "../onnx/yolov8m_640.onnx"
SAVE_PATH = BASE_DIR / "output/detection.bmp"

CONF_TH = 0.25                     # 自信度がこれ未満の検出は捨てる（下げると枠が増える）
IOU_TH = 0.45                      # 枠がこれ以上重なったら同じ物体とみなす

# デバイス、精度、AIが認識したものとマッピングする配列
DEVICE, PRECISION = {"cpu_fp32": ("CPU", "f32"),        # f32 = 細かい数値で計算（正確・遅い）
                     "gpu_fp32": ("GPU", "f32"),
                     "gpu_fp16": ("GPU", "f16"),   # f16 = 粗い数値で計算（速い）
                     "npu_fp16": ("NPU", "f16")}[MODE]

# AIが知っている80種類の物体の名前
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
COLORS = np.random.default_rng(0).integers(60, 255, size=(80, 3)).tolist()

# webカメラを開く
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: Webカメラを開けませんでした。")
    exit()

# カメラの最大解像度を要求
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)

# 解像度 W, H および FPS を取得
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None or fps > 120:
    fps = 30.0

print(f"適用されたWebカメラ解像度: {W}x{H}, FPS: {fps}")

# AIモデルを読む（前処理をモデルの中に埋め込む)
# AI は本来「0.0〜1.0 の小数に直した画像」しか受け取れない。
# その変換をパソコン側でやると毎フレーム時間がかかるので、
# 「0〜255 の画像をそのまま渡していいモデル」に作り替えてしまう。
# 変換の仕事は AI を動かすデバイス（NPU / GPU）が担当してくれる。
core = ov.Core()
print(f"使えるデバイス: {core.available_devices}")

ppp = ov.preprocess.PrePostProcessor(core.read_model(MODEL))
ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
ppp.input().model().set_layout(ov.Layout("NCHW"))
ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)

compiled = core.compile_model(ppp.build(), DEVICE,
                              {"INFERENCE_PRECISION_HINT": PRECISION,
                               "CACHE_DIR": BASE_DIR / f"../cache/ov_cache_yolo_{MODE}"})
SIZE = compiled.input(0).shape[1]      # AI が受け取る画像の大きさ（640）
print(f"検出: {MODE} ({DEVICE} / {PRECISION} / {SIZE}x{SIZE})")

# カメラ画像をAI用の正方形に収める
# 1246x1008 の画像をそのまま 640x640 に潰すと縦横比が狂って検出がずれる。
# 縦横比を保ったまま縮小し、余った上下（または左右）を灰色で埋める。
SCALE = min(SIZE / H, SIZE / W)              # 何倍に縮小するか
NH, NW = round(H * SCALE), round(W * SCALE)  # 縮小後の大きさ
TOP, LEFT = (SIZE - NH) // 2, (SIZE - NW) // 2   # 余白の幅

# AIモデルを2つ持たす
#  AIモデル用を2枚、画像も2枚用意して交互に使う。
#  1枚目の結果を待っている間に、2枚目をモデルに渡してGPU効率を高める。
requests = [compiled.create_infer_request() for _ in range(2)]
boxes_in = [np.full((1, SIZE, SIZE, 3), 114, np.uint8) for _ in range(2)]
for req, box in zip(requests, boxes_in):
    req.set_input_tensor(ov.Tensor(box, shared_memory=True))

slot = 0                                  # 次に使う箱の番号（0 と 1 を行き来する）

# 各処理の関数を定義する
def grab_frame():
    """カメラから 1 枚もらって、カラー画像の形（3色ぶん）にして返す"""
    ret, frame = cap.read()
    if not ret:
        print("カメラからのフレーム取得に失敗しました。")
        return None
    
    return frame


def send_to_ai(frame):
    """画像を AI に渡す。結果は待たずにすぐ戻ってくる（AI は裏で動き続ける）"""
    global slot
    cv2.resize(frame, (NW, NH), dst=boxes_in[slot][0, TOP:TOP + NH, LEFT:LEFT + NW])
    requests[slot].start_async()
    slot = 1 - slot                       # 次は反対側の箱を使う


def receive_result():
    """1 つ前に渡した画像の結果を受け取る（まだなら出来上がるまで待つ）"""
    req = requests[slot]                  # slotは今「古い方」を指している
    req.wait()
    return read_boxes(req.get_output_tensor(0).data)


def read_boxes(output):
    """AI の出力を、画面に描ける枠のリストに翻訳する

    output の形は (1, 84, 8400)。8400 個の枠の候補があり、1 個につき 84 個の数字。
    最初の 4 個が枠の位置（中心x, 中心y, 幅, 高さ）、残り 80 個が
    「その枠が person らしさ / bicycle らしさ / …」を表す自信度。
    """
    cand = output[0]
    best_score = cand[4:].max(0)                     # 候補ごとに一番高い自信度
    alive = np.flatnonzero(best_score > CONF_TH)     # 自信のある候補だけ残す
    alive = alive[np.argsort(-best_score[alive])[:300]]   # 多すぎると重いので上位300件
    if not len(alive):
        return []

    score = best_score[alive]
    class_id = cand[4:, alive].argmax(0)             # どの物体だと思ったか
    cx, cy, bw, bh = cand[0, alive], cand[1, alive], cand[2, alive], cand[3, alive]

    # 640x640の中の座標を、元のカメラ画像の座標に戻す（余白を引いて縮小率で割る）
    x = (cx - bw * 0.5 - LEFT) / SCALE
    y = (cy - bh * 0.5 - TOP) / SCALE
    w, h = bw / SCALE, bh / SCALE

    # 同じ物体に何個も枠が出るので、重なった枠を1個にまとめる（NMS という処理）。
    # 種類ごとにまとめたいので、種類の番号ぶん座標をずらして別物として扱わせる。
    shift = class_id * 8192
    picked = cv2.dnn.NMSBoxes(np.stack([x + shift, y + shift, w, h], 1).tolist(),
                              score.tolist(), CONF_TH, IOU_TH)
    return [(int(x[i]), int(y[i]), int(x[i] + w[i]), int(y[i] + h[i]),
             float(score[i]), int(class_id[i])) for i in np.array(picked).flatten()]


def draw_boxes(frame, boxes):
    """画像に枠と名前を描き込む"""
    for x1, y1, x2, y2, score, class_id in boxes:
        color = COLORS[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[class_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# 実行部分
print("Esc 終了 / s 保存")

frame = grab_frame()
send_to_ai(frame)                 # 助走。1枚目を先に渡しておく
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # 先に保存先のフォルダを作っておく
fps, t_prev = 0.0, time.perf_counter()

while True:
    next_frame = grab_frame()     # カメラから次の1枚をもらう
    send_to_ai(next_frame)        # それをAIに渡す（2待たずにすぐ戻る）
    boxes = receive_result()      # 1つ前の画像の結果を受け取る
    draw_boxes(frame, boxes)      # その1つ前の画像に枠を描く

    # ⑤ fps を出して画面に表示する
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / (now - t_prev)     # 数字がちらつかないよう平均をとる
    t_prev = now
    cv2.putText(frame, f"{MODE}  {fps:5.1f} fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("INFINICAM YOLOv8", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:                             # Esc : 終了
        break
    elif key == ord('s'):                     # s : 画像を保存
        cv2.imwrite(str(SAVE_PATH), frame)
        print(f"保存しました: {SAVE_PATH}")

    frame = next_frame            # さっき撮った画像が、次の周では「1つ前」になる

# ---- 後片付け ------------------------------------------------------
for req in requests:
    req.wait()
cv2.destroyAllWindows()
