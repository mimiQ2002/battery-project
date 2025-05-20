import os
import time
import cv2
import torch
from collections import Counter
from ultralytics import YOLO
from adafruit_servokit import ServoKit
from time import sleep

# --- ตั้งค่าพาธโมเดล ---
model_path = r"/home/guy/Desktop/yolo/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# --- โหลดโมเดล และกำหนด device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)
print(f"Using device: {device}")

# --- เปิดกล้อง (index 1) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam. Please check your camera connection.")

# ตั้งค่าความละเอียด
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- สร้าง ServoKit หนึ่งครั้ง ---
kit = ServoKit(channels=16)

def move_servo(servo_id, angle):
    try:
        kit.servo[servo_id].actuation_range = 180
        kit.servo[servo_id].set_pulse_width_range(500, 2500)

        if 0 <= servo_id < 16 and 0 <= angle <= 180:
            kit.servo[servo_id].angle = angle
            print(f"Servo pin {servo_id} moved to {angle} degrees.")
        else:
            print("Invalid servo pin or angle value.")
    except ValueError:
        print("Error: Please enter valid numbers for servo number and angle.")

try:
    print("Starting real-time split detection... Press Ctrl+C to stop.")
    while True:
        left_names = []
        right_names = []

        start_time = time.time()
        while time.time() - start_time < 1:  # รวบรวม detection เป็นเวลา 2 วินาที
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                continue

            h, w, _ = frame.shape
            mid = w // 2
            left_frame = frame[:, :mid]
            right_frame = frame[:, mid:]

            # ตรวจฝั่งซ้าย
            try:
                res_left = model.predict(left_frame, device=device, verbose=False)
                boxes_left = res_left[0].boxes.data.cpu().numpy()
                for det in boxes_left:
                    cls_id = int(det[5])
                    class_name = model.names.get(cls_id, f"Unknown({cls_id})")
                    left_names.append(class_name)
            except Exception as e:
                print(f"Error during left frame detection: {e}")

            # ตรวจฝั่งขวา
            try:
                res_right = model.predict(right_frame, device=device, verbose=False)
                boxes_right = res_right[0].boxes.data.cpu().numpy()
                for det in boxes_right:
                    cls_id = int(det[5])
                    class_name = model.names.get(cls_id, f"Unknown({cls_id})")
                    right_names.append(class_name)
            except Exception as e:
                print(f"Error during right frame detection: {e}")
            
        # สรุปผลการตรวจจับ
        if left_names:
            left_common = Counter(left_names).most_common(1)[0]
            print(f"Left Most Detected: {left_common[0]} ({left_common[1]} times)")
        else:
            print("Left: No detections")

        if right_names:
            right_common = Counter(right_names).most_common(1)[0]
            print(f"Right Most Detected: {right_common[0]} ({right_common[1]} times)")
        else:
            print("Right: No detections")

        # Result จาก detection
        result_left = left_common[0] if left_names else "No detection"
        result_right = right_common[0] if right_names else "No detection"

        # --- Move servo ฝั่งซ้ายตามการตรวจจับ ---
        if result_left in ["Phone", "Powerbank", "Laptop"]:
            if result_left == "Phone":
                print("Left side detected a Phone. Moving servo 0 and 1 to 60 degrees.")
                move_servo(1, 50)
                move_servo(0, 20)
                sleep(1.5)  # พักให้หมุนทัน
                print("Resetting left servos to 90 degrees.")
                move_servo(0, 125)
                move_servo(1, 50)
                sleep(1.5)  # พักให้หมุนทัน
                
            elif result_left == "Powerbank":
                print("Left side detected a Power bank. Moving servo 0 to 120 degrees and servo 1 to 60 degrees.")
                move_servo(0, 125)
                move_servo(1, 50)
                sleep(1.5)  # พักให้หมุนทัน
                print("Resetting left servos to 90 degrees.")
                move_servo(0, 125)
                move_servo(1, 50)
                sleep(1.5)  # พักให้หมุนทัน
                     
            elif result_left == "Laptop":
                print("Left side detected a Laptop. Moving servo 0 and 1 to 120 degrees.")
                move_servo(0, 120)
                move_servo(1, 165)
                sleep(1.5)  # พักให้หมุนทัน
                print("Resetting left servos to 90 degrees.")
                move_servo(1, 50)
                move_servo(0, 125)
                sleep(1.5)
            
        # --- Move servo ฝั่งขวาตามการตรวจจับ ---
        if result_right in ["phone", "Powerbank", "Laptop"]:
            if result_right == "phone":
                print("Right side detected a Phone. Moving servo 2 and 3 to 60 degrees.")
                move_servo(2, 60)
                move_servo(3, 60)
            elif result_right == "Powerbank":
                print("Right side detected a Power bank. Moving servo 2 to 120 degrees and servo 3 to 60 degrees.")
                move_servo(2, 120)
                move_servo(3, 60)
            elif result_right == "Laptop":
                print("Right side detected a Laptop. Moving servo 2 and 3 to 120 degrees.")
                move_servo(2, 120)
                move_servo(3, 120)

            sleep(1)

            # Reset กลับไปที่ 90 องศา
            print("Resetting right servos to 90 degrees.")
            move_servo(2, 90)
            move_servo(3, 90)

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting detection loop...")

finally:
    if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection loop.")
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program terminated.")
