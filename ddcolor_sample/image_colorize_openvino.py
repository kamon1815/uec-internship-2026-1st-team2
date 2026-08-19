import os
import time
import cv2
import numpy as np
import openvino as ov
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# 設定
MODE = "cpu_fp32"                              # cpu_fp32 / gpu_fp32 / gpu_fp16 / npu_fp16
MODEL = BASE_DIR / "../onnx/ddcolor_512.onnx"
INPUT = BASE_DIR / "image/image.jpg"
OUTPUT = BASE_DIR / f"output/result_{MODE}.png"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

DEVICE, PRECISION = {"cpu_fp32": ("CPU", "f32"),
                     "gpu_fp32": ("GPU", "f32"),
                     "gpu_fp16": ("GPU", "f16"),
                     "npu_fp16": ("NPU", "f16")}[MODE]

# 画像を読む
bgr = cv2.imread(INPUT)
h, w = bgr.shape[:2]

# モデルを読む
# INFERENCE_PRECISION_HINT … 計算精度。NPU は "f16" 固定
# CACHE_DIR … コンパイル結果の保存先。2 回目以降の起動が速くなる
core = ov.Core()
print(f"使えるデバイス: {core.available_devices}")
model = core.compile_model(core.read_model(MODEL), DEVICE,
                           {"INFERENCE_PRECISION_HINT": PRECISION,
                            "CACHE_DIR": BASE_DIR / f"../cache/ov_cache_ddcolor_{MODE}"})

size = model.input(0).shape[2]

# 前処理
# 元解像度の L（明るさ）を取っておく。最後にこれと色を合体させる。
orig_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]

# モデルサイズに縮小して「色みを消した画像」を作る。
# uint8 の Lab では a=b=0（無彩色）は 128 で表される。
small = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2LAB)
small[:, :, 1:] = 128
x = cv2.cvtColor(small, cv2.COLOR_LAB2RGB).transpose(2, 0, 1)[np.newaxis]
x = x.astype(np.float32) / 255.0

# 推論
model(x)                                        # ウォームアップ（初回は遅いので捨てる）

start = time.perf_counter()
ab = model(x)[0]                                # (1, 2, S, S)
elapsed_ms = (time.perf_counter() - start) * 1000
print(f"{DEVICE} / {PRECISION} : {elapsed_ms:.2f} ミリ秒 ({1000 / elapsed_ms:.1f} fps)")

# 後処理
# 小さいうちに uint8 化してから拡大する（float32 のまま拡大するより速い）
ab = np.clip(ab[0].transpose(1, 2, 0) + 128.0, 0, 255).astype(np.uint8)
ab = cv2.resize(ab, (w, h))
result = cv2.cvtColor(cv2.merge([orig_l, ab[:, :, 0], ab[:, :, 1]]),
                      cv2.COLOR_LAB2BGR)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(OUTPUT, result)
print(f"保存しました: {OUTPUT}")