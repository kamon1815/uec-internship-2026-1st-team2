import time
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# 推論を行うデバイス、モデルのパス設定
BASE_DIR = Path(__file__).resolve().parent
COLOR_MODE = "npu_fp16"                        # DDColor を動かすハード
YOLO_MODE = "gpu_fp16"                         # YOLO を動かすハード
#   どちらも cpu_fp32 / gpu_fp32 / gpu_fp16 / npu_fp16 から選ぶ

COLOR_MODEL = BASE_DIR / "../onnx/ddcolor_512.onnx"
YOLO_MODEL = BASE_DIR / "../onnx/yolov8m_640.onnx"
SAVE_PATH = BASE_DIR / "output/color_detect.bmp"

COLORIZE = True                                # 起動時にカラー化するか
CONF_TH, IOU_TH = 0.25, 0.45

MODES = {"cpu_fp32": ("CPU", "f32"), "gpu_fp32": ("GPU", "f32"),
         "gpu_fp16": ("GPU", "f16"), "npu_fp16": ("NPU", "f16")}
CLASSES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,"
    "stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,"
    "backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,"
    "baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,"
    "fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,"
    "cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,"
    "hair drier,toothbrush").split(",")
COLORS = np.random.default_rng(0).integers(60, 255, size=(80, 3)).tolist()
L_LUT = cv2.cvtColor(np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, 2),
                     cv2.COLOR_BGR2LAB)[0, :, 0].copy()      # グレー値→LabのL

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

# モデル
core = ov.Core()
print(f"使えるデバイス: {core.available_devices}")


class AsyncModel:
    """1つのモデルを2本の依頼で交互に回す。
    1本目の処理が走っている間に、2本目の処理に次の画像を書き込む。"""

    def __init__(self, path, mode, name):
        device, precision = MODES[mode]
        ppp = ov.preprocess.PrePostProcessor(core.read_model(path))
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
        m = core.compile_model(ppp.build(), device,
                               {"INFERENCE_PRECISION_HINT": precision,
                                "CACHE_DIR": BASE_DIR / f"../cache/ov_cache_{name}_{mode}"})
        self.size = m.input(0).shape[1]
        self.reqs = [m.create_infer_request() for _ in range(2)]
        self.bufs = [np.full((1, self.size, self.size, 3), 114, np.uint8) for _ in range(2)]
        for r, b in zip(self.reqs, self.bufs):
            r.set_input_tensor(ov.Tensor(b, shared_memory=True))
        self.slot = 0
        print(f"{name}: {mode} ({device}/{precision}/{self.size}px)")

    def buffer(self):                    # 次に送る画像の書き込み先
        return self.bufs[self.slot][0]

    def send(self):                      # 投げる。結果は待たずにすぐ戻る
        self.reqs[self.slot].start_async()
        self.slot = 1 - self.slot

    def receive(self):                   # 1つ前に投げた画像の結果を受け取る
        self.reqs[self.slot].wait()
        return self.reqs[self.slot].get_output_tensor(0).data

    def drain(self):                     # 走っている推論を回収して最初の状態に戻す
        for r in self.reqs:
            r.wait()
        self.slot = 0


ddcolor = AsyncModel(COLOR_MODEL, COLOR_MODE, "ddcolor")
yolo = AsyncModel(YOLO_MODEL, YOLO_MODE, "yolo")

# YOLO用レターボックスの寸法（カメラ解像度は固定なので1回だけ計算する）
SCALE = min(yolo.size / H, yolo.size / W)
NH, NW = round(H * SCALE), round(W * SCALE)
TOP, LEFT = (yolo.size - NH) // 2, (yolo.size - NW) // 2
ab_full = np.empty((H, W, 2), np.uint8)               # 合成で使い回す


def grab_gray():
    """カメラから 1 枚もらって、カラー画像の形（3色ぶん）にして返す"""
    ret, frame = cap.read()
    if not ret:
        print("カメラからのフレーム取得に失敗しました。")
        return None
    return frame[:, :, 0] if frame.ndim == 3 else frame


def merge_color(gray, ab):
    """AI が出した小さい色(a,b) と、元解像度の明るさ(L) を合体させる。
    L の変換は cv2.LUT を使う（numpy の L_LUT[gray] は同じ結果だが 3.5ms、こちらは 0.24ms）。"""
    small = np.clip(ab[0].transpose(1, 2, 0) + 128.0, 0, 255).astype(np.uint8)
    cv2.resize(small, (W, H), dst=ab_full)
    lab = cv2.merge([cv2.LUT(gray, L_LUT), ab_full[:, :, 0], ab_full[:, :, 1]])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def draw_detections(frame, output):
    """YOLO の出力 (1, 84, N) から枠を選んで、そのまま描く。
    84 = 中心x, 中心y, 幅, 高さ + 80 クラスぶんの自信度。"""
    cand = output[0]
    best = cand[4:].max(0)
    alive = np.flatnonzero(best > CONF_TH)
    alive = alive[np.argsort(-best[alive])[:300]]     # 候補が多すぎるとNMSが重くなる
    if not len(alive):
        return
    score, cid = best[alive], cand[4:, alive].argmax(0)
    w, h = cand[2, alive] / SCALE, cand[3, alive] / SCALE
    x = (cand[0, alive] - cand[2, alive] * 0.5 - LEFT) / SCALE   # レターボックスを外す
    y = (cand[1, alive] - cand[3, alive] * 0.5 - TOP) / SCALE
    shift = cid * 8192                                # クラスごとにNMSするためのずらし
    for i in np.array(cv2.dnn.NMSBoxes(np.stack([x + shift, y + shift, w, h], 1).tolist(),
                                       score.tolist(), CONF_TH, IOU_TH)).flatten():
        x1, y1, color = int(x[i]), int(y[i]), COLORS[cid[i]]
        cv2.rectangle(frame, (x1, y1), (int(x[i] + w[i]), int(y[i] + h[i])), color, 2)
        label = f"{CLASSES[cid[i]]} {score[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ---- 本体 ----------------------------------------------------------
# ベルトコンベアのように、工程ごとに違うフレームを同時に流す。
# カラー:取得 →[カラー化]→ 合成 →[検出]→表示（表示は2コマ遅れ）
# モノクロ:取得─────────────→[検出]→表示（表示は1コマ遅れ）
# 検出にかけるのは常に「画面に出る画像そのもの」なので、枠は必ず被写体に合う。
print("Esc 終了 / s 保存 / c カラー / モノクロ切り替え")
colorize = COLORIZE
prev_gray = None          # カラー化に投げた画像のモノクロ版（L成分に使う）
pending = None            # 検出に投げた画像（枠を描く相手）
frame = None
fps, t_prev = 0.0, time.perf_counter()

while True:
    gray = grab_gray()                                        # カメラから1枚

    if colorize:
        cv2.cvtColor(cv2.resize(gray, (ddcolor.size, ddcolor.size)),
                     cv2.COLOR_GRAY2BGR, dst=ddcolor.buffer())
        ddcolor.send()                                        # カラー化を投げる
        # 1つ前のカラー化結果を受け取って合成する（待つ間もう一方は検出中）
        frame = merge_color(prev_gray, ddcolor.receive()) if prev_gray is not None else None
        prev_gray = gray
    else:
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if frame is not None:                                     # 出来た画像を検出に投げる
        cv2.resize(frame, (NW, NH), dst=yolo.buffer()[TOP:TOP + NH, LEFT:LEFT + NW])
        yolo.send()

    if pending is not None:                                   # 1つ前の検出結果を描く
        draw_detections(pending, yolo.receive())
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - t_prev)
        t_prev = now
        cv2.putText(pending, f"{'color' if colorize else 'mono'}  {fps:5.1f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("INFINICAM", pending)                      # 画面に出す

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s') and pending is not None:
        SAVE_PATH.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(SAVE_PATH), pending)
        print(f"保存しました: {SAVE_PATH}")
    elif key == ord('c'):                                     # カラー/モノクロ切り替え
        colorize = not colorize
        for m in (ddcolor, yolo):                             # 走っている推論を回収して
            m.drain()                                         # パイプラインを張り直す
        prev_gray = frame = None
    pending = frame

for m in (ddcolor, yolo):
    m.drain()
cv2.destroyAllWindows()