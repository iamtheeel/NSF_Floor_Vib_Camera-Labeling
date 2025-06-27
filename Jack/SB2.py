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

from OCR_Detect import timeWith_ms

# === MODEL PATH ===
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_lite.task" #5.5 MB
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_full.task" #9.0 MB
model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_heavy.task" #29.2 MB

# === VIDEO FILE ===
dir = r'C:\Users\notyo\Documents\STARS\StudentData\25-06-26'
file = 'jump_3_6-26-2025_1-38-57 PM.asf'
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
clipRunTime_s = 0
clipStartTime_s = 0
clipStartFrame = 0
clipRunFrames = int((fCount - clipStartFrame) if clipRunTime_s == 0 else (clipRunTime_s * fps))


prev_time = None
first_rollover_time = None  # To store the OCR time string
first_rollover_frame = None
first_rollover_timestamp_ms = None
first_rollover_detected = False
last_rollover_ocr_time = None
ms_since_last_rollover = 0
frames_since_rollover = 0

display_times = []

videoObject.set(cv2.CAP_PROP_POS_MSEC, clipStartTime_s * 1000)

# === CSV SETUP ===
csv_path = r"C:\Users\notyo\Documents\STARS\NSF_Floor_Vib_Camera-Labeling\NSF_Floor_Vib_Camera-Labeling\Jack\trialData\boing_heel_tracking.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "LeftHeel_m", "RightHeel_m"])
    print("Saving CSV to:", os.path.abspath(csv_path))

timeRetrieval = timeWith_ms(frameTime_ms)

# === Frame Loop ===
for i in range(clipRunFrames):
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms)
    success, frame = videoObject.read()
    if not success:
        print("⚠️ Frame read failure")
        break

    current_time = getDateTime(frame)  # or getTime(frame) if you want just the time

    timeOutput_ms_str = timeRetrieval.calc_ms(current_time, i, True)

    display_times.append(timeOutput_ms_str)    


    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if len(pose_landmarker_result.pose_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        landmarks_w = pose_landmarker_result.pose_world_landmarks[0]

        # Draw landmarks
        drawLandmark(frame, landmarks[29], [255, 0, 0])    # Left heel
        drawLandmark(frame, landmarks[30], [0, 255, 0])    # Right heel
        
        print(f"Foot position | left heel: {(landmarks[29].y)*h:.0f}, right heel: {(landmarks[30].y)*h:.0f}")


                
        left_heel_dist_loop = find_dist_from_y(landmarks[29].y*h)
        right_heel_dist_loop = find_dist_from_y(landmarks[30].y*h)

        # Save to CSV
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timeOutput_ms_str,                # The OCR time + ms string
                left_heel_dist_loop,        # Left heel position in pixels
                right_heel_dist_loop          # Right heel position in pixels
            ])


    # Show frame
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)

    key = cv2.waitKey(1)
    if key == ord('q') & 0xFF:
        break

#frames = []
display_times = []
left_heel = []
right_heel = []

left_heel_dist = [find_dist_from_y(yl * h) for yl in left_heel]
right_heel_dist = [find_dist_from_y(yr * h) for yr in right_heel]


with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        display_times.append(row["Time"])
        left_heel_dist.append(float(row["LeftHeel_m"]))
        right_heel_dist.append(float(row["RightHeel_m"]))



N = 10  # Show every 10th label
plt.figure(figsize=(10, 5))
plt.plot(display_times, left_heel_dist, label="Left Heel Distance", color='blue')
plt.plot(display_times, right_heel_dist, label="Right Heel Distance", color='green')
plt.xlabel("OCR Time + ms")
plt.ylabel("Heel Distance (m)")
plt.title("Heel Distance Over Time")
plt.legend()
plt.tight_layout()
plt.xticks(
    ticks=range(0, len(display_times), N),
    labels=[display_times[i] for i in range(0, len(display_times), N)],
    rotation=45
)
plt.show()


videoObject.release()
cv2.destroyAllWindows()