import time
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# 推論を行うデバイス、モデルのパス設定
BASE_DIR = Path(__file__).resolve().parent
MODE = "cpu_fp32"                              # cpu_fp32 / gpu_fp32 / gpu_fp16 / npu_fp16
MODEL = BASE_DIR / "../onnx/ddcolor_512.onnx"
SAVE_PATH = BASE_DIR / "output/colorized.bmp"                    # sキーで保存する先

DEVICE, PRECISION = {"cpu_fp32": ("CPU", "f32"),
                     "gpu_fp32": ("GPU", "f32"),
                     "gpu_fp16": ("GPU", "f16"),
                     "npu_fp16": ("NPU", "f16")}[MODE]

# グレー値 → Lab の L(uint8) 変換表。モノクロ入力なら誤差ゼロで、
L_LUT = cv2.cvtColor(np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, 2),
                     cv2.COLOR_BGR2LAB)[0, :, 0].copy()

# カメラを開く（シャッター速度100固定)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: Webカメラを開けませんでした。")
    exit()

# カラー化モデルを読む
core = ov.Core()
model = core.compile_model(core.read_model(MODEL), DEVICE,
                           {"INFERENCE_PRECISION_HINT": PRECISION,
                            "CACHE_DIR": BASE_DIR / f"../cache/ov_cache_ddcolor_{MODE}"})
size = model.input(0).shape[2]
print(f"カラー化: {MODE}  ({DEVICE} / {PRECISION} / {size}x{size})")
print("Esc 終了 / s 保存 / c カラー化 ON-OFF")

colorize = True
fps, t_prev = 0.0, time.perf_counter()

while True:
    # 1フレーム取得してデコード（ここまで元のサンプルと同じ）
    ret, gray = cap.read()
    if not ret:
        print("カメラからのフレーム取得に失敗しました。")
        break
    if gray.ndim == 3:                          # 1chに揃える
        gray = gray[:, :, 0]
    h, w = gray.shape[:2]

    if colorize:
        # 前処理:縮小→3chに複製→0..1 のfloat32に変換
        small = cv2.resize(gray, (size, size))
        x = np.repeat(small[None, None], 3, axis=1).astype(np.float32) / 255.0

        # 推論:a,bを予測
        ab = model(x)[0]                                    # (1, 2, size, size)

        # 後処理:小さいうちに uint8 化してから拡大
        ab = np.clip(ab[0].transpose(1, 2, 0) + 128.0, 0, 255).astype(np.uint8)
        ab = cv2.resize(ab, (w, h))
        lab = cv2.merge([L_LUT[gray], ab[:, :, 0], ab[:, :, 1]])   # Lは元解像度
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 表示
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / (now - t_prev)      # 移動平均でちらつきを抑える
    t_prev = now
    cv2.putText(frame, f"{MODE}  {fps:5.1f} fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("INFINICAM", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:                                # Esc:終了
        break
    elif key == ord('s'):                        # s:保存
        cv2.imwrite(SAVE_PATH, frame)
        print(f"保存しました: {SAVE_PATH}")
    elif key == ord('c'):                        # c:カラー化 ON/OFF
        colorize = not colorize

cv2.destroyAllWindows()
cap.release()
