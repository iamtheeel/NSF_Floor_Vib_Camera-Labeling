####
#   Jack Capito
#   STARS Summer 2025
#   Dr J Lab
###
# Pose tracking with MediaPipe and OCR + CSV output
####

# === Imports ===
import time
import math
import matplotlib.pyplot as plt
import csv

import sys
import os
sys.path.append(os.path.abspath('...'))

from distance_position import find_dist_from_y

import cv2  # pip install opencv-python
import numpy as np
import pytesseract  # pip install pytesseract to do OCR

import mediapipe as mp  # pip install mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === MODEL PATH ===
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_lite.task" #5.5 MB
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_full.task" #9.0 MB
model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_heavy.task" #29.2 MB

# === VIDEO FILE ===
dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_06_11'
file = 'subject_2_test_5_6-11-2025_5-54-26 PM.asf'
fileName = f"{dir}/{file}"  # Path to the video file
print(fileName)

# === Open video ===
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = 30
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
        time_str = data['text'][5]
        return f"{time_str}"
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
frameTime_ms = 1000/30 #How long of a time does each frame cover, convert from seconds to milliseconds / a.k.a. frame rate 

# === Clip Setup ===
clipRunTime_s = 20
clipStartTime_s = 10
clipStartFrame = 0
clipRunFrames = int((fCount - clipStartFrame) if clipRunTime_s == 0 else (clipRunTime_s * fps))


prev_time = None
frames_since_last = 0
first_rollover_time = None
first_rollover_frame = None
first_rollover_timestamp_ms = None
first_rollover_detected = False
last_rollover_ocr_time = None
ms_since_last_rollover = 0
frames_since_rollover = 0

display_times = []

videoObject.set(cv2.CAP_PROP_POS_MSEC, clipStartTime_s * 1000)

# === CSV SETUP ===
csv_path = "heel_tracking_output.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Display_Time", "LeftHeel_Y", "RightHeel_Y"])

# === Frame Loop ===
for i in range(clipRunFrames):
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms)
    success, frame = videoObject.read()
    if not success:
        print("⚠️ Frame read failure")
        break

    current_time = getDateTime(frame)  # or getTime(frame) if you want just the time
    display_time = current_time

    if prev_time is None:
        prev_time = current_time

    if current_time != prev_time:
        if not first_rollover_detected:
            first_rollover_detected = True
            last_rollover_ocr_time = current_time
            ms_since_last_rollover = 0
            frames_since_rollover = 0
            print(f"First OCR rollover at frame {i}, OCR time: {current_time}")
        else:
            last_rollover_ocr_time = current_time
            ms_since_last_rollover = 0
            frames_since_rollover = 0
        prev_time = current_time

    frames_since_rollover += 1
    ms_since_last_rollover = ((frames_since_rollover - 1) * frameTime_ms) % 1000

    # After the first rollover, display the OCR time plus ms since last rollover
    if first_rollover_detected:
        # Format: OCR time + ms since last rollover
        display_time = f"{last_rollover_ocr_time}.{int(ms_since_last_rollover):03d}"
        print(f"Vid time: {display_time}")
        display_times.append(display_time)
    else:
        display_times.append("")


    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if len(pose_landmarker_result.pose_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        landmarks_w = pose_landmarker_result.pose_world_landmarks[0]

        # Draw landmarks
        drawLandmark(frame, landmarks[29], [255, 0, 0])    # Left heel
        drawLandmark(frame, landmarks[30], [0, 255, 0])    # Right heel
        
        print(f"Foot position | left heel: {(landmarks[29].y)*h:.0f}, right heel: {(landmarks[30].y)*h:.0f}")

        # Save to CSV
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                display_time,                # The OCR time + ms string
                landmarks[29].y * h,         # Left heel position in pixels
                landmarks[30].y * h          # Right heel position in pixels
            ])



    # Show frame
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)

    key = cv2.waitKey(1)
    if key == ord('q') & 0xFF:
        break

frames = []
left_heel = []
right_heel = []

with open("heel_tracking_output.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        frames.append(int(row["Frame"]))
        left_heel.append(float(row["LeftHeel_Y"]))
        right_heel.append(float(row["RightHeel_Y"]))
yl_conv = (f"{(landmarks[29].y):.0f}")
yr_conv = (f"{(landmarks[30].y):.0f}")

left_heel_dist = [find_dist_from_y(yl * h) for yl in left_heel]
right_heel_dist = [find_dist_from_y(yr * h) for yr in right_heel]

N = 10  # Show every 10th label
plt.figure(figsize=(10, 5))
plt.plot(display_times, left_heel_dist, label="Left Heel Distance", color='blue')
plt.plot(display_times, right_heel_dist, label="Right Heel Distance", color='green')
plt.xlabel("OCR Time + ms")
plt.ylabel("Heel Distance (m)")
plt.title("Heel Distance Over Time")
plt.legend()
plt.tight_layout()

# Set x-ticks to every Nth label
plt.xticks(ticks=range(0, len(display_times), N), labels=[display_times[i] for i in range(0, len(display_times), N)], rotation=45)

plt.show()

videoObject.release()
cv2.destroyAllWindows()
