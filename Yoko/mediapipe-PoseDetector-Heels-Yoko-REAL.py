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
import sys
import os

import cv2
import numpy as np
import pytesseract

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === Fix import path to reach distance_position.py ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_position import find_dist_from_y  # ✅ Import your custom function

# === MODEL PATH ===
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task"
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"

# === VIDEO FILE ===
#video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/2025_STARS_ProfJ/StudentData/25_06_11'
#video_file = 'subject_2_test_1_6-11-2025_5-40-27 PM.asf'
#video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/25_06_18/Subject_1'
#video_file = 'Sub_1_Run_1_6-18-2025_11-45-46 AM.asf'
#fileName = f"{video_dir}/{video_file}"
video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/25_06_18/Subject_1'
#video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/25_06_18/Subject_2'
#video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/25_06_18/Subject_3'
video_file = 'Sub_1_Run_1_6-18-2025_11-45-46 AM.asf'
#video_file = 'Sub_1_Run_2__6-18-2025_11-47-57 AM.asf'
#video_file = 'Sub_1_Run_3__6-18-2025_11-49-29 AM.asf'
#video_file = 'Sub_1_Run_4__6-18-2025_11-50-26 AM.asf'
#video_file = 'sub_2_run_1_6-18-2025_11-36-03 AM.asf'
#video_file = 'sub_2_run_3_pt_1_6-18-2025_11-40-17 AM.asf'
#video_file = 'sub_2_run_3_pt_2_6-18-2025_11-39-54 AM.asf' #####ISSUE WITH FRAME READ#####
#video_file = 'sub_2_run_4_6-18-2025_11-41-35 AM.asf'
#video_file = 'sub_2_run_5_6-18-2025_11-42-48 AM.asf'
#video_file = 'sub_3_run_4_F_6-18-2025_11-26-08 AM.asf'
#video_file = 'sub3_run5_6-18-2025_11-28-28 AM.asf'
#video_file = 'Sub3_run6_6-18-2025_11-32-05 AM.asf'
#video_file = 'Sub3_run7_6-18-2025_11-34-22 AM.asf'


fileName = f"{video_dir}/{video_file}"

model_path = r"C:\Users\smitt\STARS\pose_landmarker_heavy.task" # 29.2 MiB
#Video File
dir = r"E:\STARS\day1_data"
file = r"25_06_03_s1_1.asf"
fileName = f"{dir}/{file}"
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

# === CSV SETUP (✅ correct position: outside loop) ===
csv_path = r"E:\STARS\myfile2.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Frame", "Timestamp_ms",
        "LeftHeel_Y", "RightHeel_Y",
        "LeftHeel_Distance", "RightHeel_Distance"
    ])

# === Frame Timing ===
frameTime_ms = 1000 / fps

# === Clip Setup ===
clipRunTime_s = 0
clipStartTime_s = 60
clipRunFrames = int(fCount - (clipStartTime_s * fps)) if clipRunTime_s == 0 else int(clipRunTime_s * fps)

videoObject.set(cv2.CAP_PROP_POS_MSEC, clipStartTime_s * 1000)

# === Frame Loop ===
for i in range(clipRunFrames):
    frame_timestamp_ms = frameTime_ms * i
    success, frame = videoObject.read()
    if not success:
        print("⚠️ Frame read failure")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))

    if len(pose_landmarker_result.pose_landmarks) > 0:
        adjusted_time_ms = int(frame_timestamp_ms)

        landmarks = pose_landmarker_result.pose_landmarks[0]
        landmarks_w = pose_landmarker_result.pose_world_landmarks[0]

        # === Heel Y values (normalized and pixel)
        left_heel_y_norm = landmarks[29].y
        right_heel_y_norm = landmarks[30].y
        left_heel_y_px = left_heel_y_norm * h
        right_heel_y_px = right_heel_y_norm * h

        # === Distances using your function
        left_dist = find_dist_from_y(left_heel_y_px, debug=True)
        right_dist = find_dist_from_y(right_heel_y_px, debug=True)

        # === Print
        print(f"Left heel Y: {left_heel_y_norm:.4f} → {left_dist:.4f} m | Right heel Y: {right_heel_y_norm:.4f} → {right_dist:.4f} m")
        print(f"frametime: {adjusted_time_ms:.0f} ms")

        # === Save to CSV (✅ append only)
        csv_path = "/Volumes/MY PASSPORT/SFSU_STARS/"
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                i,
                adjusted_time_ms,
                left_heel_y_norm,
                right_heel_y_norm,
                left_dist,
                right_dist
            ])

        # === Draw landmarks
        def drawLandmark(frame, landmark, color=(0, 0, 255)):
            center = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, center, 6, color, -1)

        def drawLine(frame, lm1, lm2, color):
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(frame, pt1, pt2, color, 3)

        drawLandmark(frame, landmarks[29], [255, 0, 0])    # Left heel
        drawLandmark(frame, landmarks[30], [0, 255, 0])    # Right heel
        drawLandmark(frame, landmarks[23], [255, 255, 0])  # Left hip
        drawLandmark(frame, landmarks[24], [0, 255, 255])  # Right hip
        drawLine(frame, landmarks[29], landmarks[23], (255, 100, 100))
        drawLine(frame, landmarks[30], landmarks[24], (100, 255, 100))

    # === Show frame
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)

    key = cv2.waitKey(1)
    if key == ord('q') & 0xFF:
        break

videoObject.release()
cv2.destroyAllWindows()
