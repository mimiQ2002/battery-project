import cv2
import torch
import time  # Import time module for delay
from ultralytics import YOLO

# Correct the path to your custom model
model_path = r"E:\Project S\YOLOv11\runs\detect\train22\weights\best.pt"  # Use "last.pt" if best.pt is missing

# Debug: Check if the model file exists
import os
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
print(f"Model file found: {model_path}")

# Load the trained YOLO model
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# Check if GPU is available and use it for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

# Open webcam (0 = default camera, 1 = external camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam. Ensure it is connected and accessible.")
print("Webcam opened successfully.")

# Set webcam resolution (optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
print("Webcam resolution set to 640x480.")

# Real-time detection loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting...")
            break

        # Split the frame into left and right halves
        height, width, _ = frame.shape
        mid = width // 2
        left_frame = frame[:, :mid]
        right_frame = frame[:, mid:]

        # Run YOLO model on the left frame
        try:
            left_results = model(left_frame)
            left_detections = left_results[0].boxes.data.cpu().numpy()  # Convert to NumPy array
            print("Left Frame Detections:")
            print("    xmin    ymin    xmax   ymax  confidence  class    name")
            for detection in left_detections:
                xmin, ymin, xmax, ymax, confidence, cls = detection[:6]
                name = model.names[int(cls)]  # Get class name
                print(f"{xmin:8.2f} {ymin:8.2f} {xmax:8.2f} {ymax:8.2f} {confidence:11.6f} {int(cls):6} {name}")
            left_annotated = left_results[0].plot()
        except Exception as e:
            print(f"Error during left frame processing: {e}")
            left_annotated = left_frame

        # Run YOLO model on the right frame
        try:
            right_results = model(right_frame)
            right_detections = right_results[0].boxes.data.cpu().numpy()  # Convert to NumPy array
            print("Right Frame Detections:")
            print("    xmin    ymin    xmax   ymax  confidence  class    name")
            for detection in right_detections:
                xmin, ymin, xmax, ymax, confidence, cls = detection[:6]
                name = model.names[int(cls)]  # Get class name
                print(f"{xmin:8.2f} {ymin:8.2f} {xmax:8.2f} {ymax:8.2f} {confidence:11.6f} {int(cls):6} {name}")
            right_annotated = right_results[0].plot()
        except Exception as e:
            print(f"Error during right frame processing: {e}")
            right_annotated = right_frame

        # Display the annotated left and right frames
        cv2.imshow("Left Frame Detection", left_annotated)
        cv2.imshow("Right Frame Detection", right_annotated)

        # Add a delay of 0.5 seconds
        time.sleep(0.5)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection loop.")
            break
except Exception as e:
    print(f"Unexpected error during detection loop: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program terminated.")