import os
import time
import cv2
import torch
from collections import Counter
from ultralytics import YOLO

# --- ตั้งค่าพาธโมเดล ---
model_path = r"E:\Project S\YOLOv11\runs\detect\train22\weights\best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# --- โหลดโมเดล และกำหนด device ---
model = YOLO(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# --- เปิดกล้อง ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    print("Starting real-time split detection... Press Ctrl+C to stop.")
    while True:
        # เตรียม list เก็บชื่อฝั่งซ้าย–ขวา
        left_names = []
        right_names = []

        # เริ่มนับเวลาสำหรับ window 2 วินาที
        window_start = time.time()
        while time.time() - window_start < 2:
            ret, frame = cap.read()
            if not ret:
                continue

            # แบ่งภาพซ้าย–ขวา
            h, w, _ = frame.shape
            mid = w // 2
            left_frame  = frame[:, :mid]
            right_frame = frame[:, mid:]

            # ตรวจจับฝั่งซ้าย
            try:
                res_left = model(left_frame)
                boxes_left = res_left[0].boxes.data.cpu().numpy()
                for det in boxes_left:
                    cls_id = int(det[5])
                    left_names.append(model.names[cls_id])
            except Exception as e:
                print("Left detect error:", e)

            # ตรวจจับฝั่งขวา
            try:
                res_right = model(right_frame)
                boxes_right = res_right[0].boxes.data.cpu().numpy()
                for det in boxes_right:
                    cls_id = int(det[5])
                    right_names.append(model.names[cls_id])
            except Exception as e:
                print("Right detect error:", e)

        # สรุปผลฝั่งซ้าย
        if left_names:
            most_left, cnt_left = Counter(left_names).most_common(1)[0]
            print(f"🔍 Left side: {most_left} ({cnt_left} times)")
        else:
            print("❌ Left side: No detections")

        # สรุปผลฝั่งขวา
        if right_names:
            most_right, cnt_right = Counter(right_names).most_common(1)[0]
            print(f"🔍 Right side: {most_right} ({cnt_right} times)")
        else:
            print("❌ Right side: No detections")

        # รีเซ็ตและวนใหม่อัตโนมัติ

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program terminated.")
