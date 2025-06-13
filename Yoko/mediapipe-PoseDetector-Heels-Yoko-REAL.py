####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# Pose tracking with MediaPipe and OCR + CSV output
####

# === Imports ===
import time
import math
import csv

import cv2  # pip install opencv-python
import numpy as np
import pytesseract  # pip install pytesseract to do OCR

import mediapipe as mp  # pip install mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === MODEL PATH ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"

# === VIDEO FILE ===
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

# === Open video ===
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = videoObject.get(cv2.CAP_PROP_FPS)
fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT)
w = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))

dispFact = 2
displayRez = (int(w / dispFact), int(h / dispFact))

# === MediaPipe Setup ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=True
)
landmarker = PoseLandmarker.create_from_options(options)

# === OCR timestamp function ===
def getDateTime(frame):
    dateTime_img = frame[0:46, 0:384, :]
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)
    dateTime_img_bw = 255 - dateTime_img_bw
    data = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    try:
        date_str = data['text'][4]
        time_str = data['text'][5]
        return f"{date_str} {time_str}"
    except:
        return "OCR Error"

def drawLandmark(frame, landmark, color=(0, 0, 255)):
    center = (int(landmark.x * w), int(landmark.y * h))
    cv2.circle(frame, center, 6, color, -1)

def drawLine(frame, lm1, lm2, color):
    pt1 = (int(lm1.x * w), int(lm1.y * h))
    pt2 = (int(lm2.x * w), int(lm2.y * h))
    cv2.line(frame, pt1, pt2, color, 3)

def calc_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# === Frame Timing (manual override) ===
frameTime_ms = 33000  # 1 frame every 33 seconds

# === Clip Setup ===
clipRunTime_s = 0
clipStartTime_s = 30
clipStartFrame = 0
clipRunFrames = int((fCount - clipStartFrame) if clipRunTime_s == 0 else (clipRunTime_s * fps))

videoObject.set(cv2.CAP_PROP_POS_MSEC, clipStartTime_s * 1000)

# === CSV SETUP ===
csv_path = "heel_tracking_output.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Timestamp_ms", "LeftHeel_Y", "RightHeel_Y"])

# === Frame Loop ===
for i in range(clipRunFrames):
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms)
    success, frame = videoObject.read()
    if not success:
        print("⚠️ Frame read failure")
        break

    timestamp_str = getDateTime(frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if len(pose_landmarker_result.pose_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        landmarks_w = pose_landmarker_result.pose_world_landmarks[0]

        # Draw landmarks
        drawLandmark(frame, landmarks[29], [255, 0, 0])    # Left heel
        drawLandmark(frame, landmarks[30], [0, 255, 0])    # Right heel
        drawLandmark(frame, landmarks[23], [255, 255, 0])  # Left hip
        drawLandmark(frame, landmarks[24], [0, 255, 255])  # Right hip

        drawLine(frame, landmarks[29], landmarks[23], (255, 100, 100))
        drawLine(frame, landmarks[30], landmarks[24], (100, 255, 100))

        strideLen = calc_dist(landmarks_w[29], landmarks_w[30])
        left_leg_len = calc_dist(landmarks_w[23], landmarks_w[29])
        right_leg_len = calc_dist(landmarks_w[24], landmarks_w[30])

        print(f"Frame timestamp: {frame_timestamp_ms} ms | left heel: {landmarks[29].y}, right heel: {landmarks[30].y}")

        # Save to CSV
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                i,
                frame_timestamp_ms,
                landmarks[29].y,
                landmarks[30].y
            ])

        # Optional: show segmentation mask
        if pose_landmarker_result.segmentation_masks is not None:
            mask = pose_landmarker_result.segmentation_masks[0].numpy_view()
            cv2.imshow("Seg mask", mask)

    # Show frame
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)

    key = cv2.waitKey(1)
    if key == ord('q') & 0xFF:
        break

videoObject.release()
cv2.destroyAllWindows()
