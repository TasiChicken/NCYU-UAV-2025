import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords  # 用一般的 NMS
from utils.plots import plot_one_box

# ----------------------------
# 設定檔案路徑
# ----------------------------
WEIGHT = './best.pt'
INPUT_VIDEO = './demo_src.mp4'              # <<== 你的輸入影片路徑
OUTPUT_VIDEO = './demo_output.mp4'     # <<== 輸出影片名稱

# ----------------------------
# 載入模型
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 如果遇到跟 train 一樣的 torch.load 安全性錯誤，
# 記得去 models/experimental.py 裡把 torch.load(...) 改成 weights_only=False
model = attempt_load(WEIGHT, map_location=device)

if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)

model.eval()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# ----------------------------
# 開啟輸入影片 & 設定輸出影片
# ----------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"[Error] 無法開啟影片: {INPUT_VIDEO}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 編碼器：mp4v / XVID / avc1 都可以，看助教環境
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"[Info] 開始處理影片...")
print(f"       輸入: {INPUT_VIDEO}")
print(f"       輸出: {OUTPUT_VIDEO}")
print(f"       解析度: {width}x{height}, FPS: {fps}")

# ----------------------------
# 逐 frame 讀取、偵測、畫框、寫出
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 影片讀完

    img_orig = frame.copy()

    # --- 前處理：letterbox resize ---
    # 這裡用 640x640，stride 用模型本身的 stride
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
    img = letterbox(img_orig, (640, 640), stride=stride, auto=True)[0]

    # HWC -> Tensor [1, C, H, W]
    if device == "cuda":
        img = transforms.ToTensor()(img).to(device).half().unsqueeze(0)
    else:
        img = transforms.ToTensor()(img).to(device).float().unsqueeze(0)

    # --- 推論 ---
    with torch.no_grad():
        pred = model(img)[0]

    # --- NMS ---
    # conf_thres=0.25, iou_thres=0.45 可自行調整
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        # 把 bbox 從 640x640 座標 scale 回原始 frame 大小
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_orig.shape).round()

        # --- 畫框 + Label + Confidence ---
        for *xyxy, conf, cls in pred:
            cls_id = int(cls)
            label = f'{names[cls_id]} {conf:.2f}'
            plot_one_box(
                xyxy,
                img_orig,
                label=label,
                color=colors[cls_id],
                line_thickness=2
            )

    # 寫入輸出影片
    out.write(img_orig)

    # 若想邊看邊跑可以打開這段
    # cv2.imshow("Detected", img_orig)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# ----------------------------
# 收尾
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("[Info] 處理完成 ✅")
