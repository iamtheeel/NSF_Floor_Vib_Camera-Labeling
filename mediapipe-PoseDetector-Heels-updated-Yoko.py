####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking with 3D World Coordinates, OCR Timestamp,
# Stride Estimation, Heel Status (Moving/Still), and CSV Logging
####

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import pytesseract
import csv

# === MODEL SETUP ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"
video_file = "/Volumes/MY PASSPORT/Stars_day1Data/s2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf"

video = cv2.VideoCapture(video_file)
if not video.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
frameTime_ms = 1000 / fps
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === LANDMARKER OPTIONS ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

# === FRAME RANGE ===
startSecs = 125
endSecs = 155
start_frame = int(fps * startSecs)
end_frame = int(fps * endSecs)
num_frames = end_frame - start_frame
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# === OCR FUNCTION ===
def getDateTime(frame):
    roi = frame[0:45, 0:384]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    data = pytesseract.image_to_data(inverted, output_type=pytesseract.Output.DICT)
    try:
        return f"{data['text'][4]} {data['text'][5]}"
    except:
        return "Unknown Time"

# === CSV FILES ===
output_csv = "stride_data_world.csv"
stride_event_csv = "stride_events_world.csv"
with open(output_csv, 'w', newline='') as f:
    csv.writer(f).writerow(["Frame", "Timestamp", "Stride_m", "Stride_ft", "Planted_Leg", "LeftHeelStatus", "RightHeelStatus"])
with open(stride_event_csv, 'w', newline='') as f:
    csv.writer(f).writerow(["Frame", "Timestamp", "Stride_m", "Stride_ft", "Planted_Leg", "Stationary_Heel"])

# === TRACKING VARIABLES ===
prev_lheel = prev_rheel = None
stationary_threshold_m = 0.01

# === POSE LANDMARK PROCESSING ===
with PoseLandmarker.create_from_options(options) as detector:
    frame_idx = 0
    while frame_idx < num_frames:
        success, frame = video.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp_ms = int((start_frame + frame_idx) * frameTime_ms)
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        timestamp_str = getDateTime(frame)

        if result.pose_landmarks and result.pose_world_landmarks:
            lm2d = result.pose_landmarks[0]
            lm3d = result.pose_world_landmarks[0]

            def get_px(lm): return int(lm.x * w), int(lm.y * h)
            lx, ly = get_px(lm2d[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value])
            rx, ry = get_px(lm2d[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value])
            lhx, lhy = get_px(lm2d[mp.solutions.pose.PoseLandmark.LEFT_HIP.value])
            rhx, rhy = get_px(lm2d[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value])

            lheel = lm3d[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value]
            rheel = lm3d[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value]
            lhip = lm3d[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            rhip = lm3d[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            lheel_pos = (lheel.x, lheel.y, lheel.z)
            rheel_pos = (rheel.x, rheel.y, rheel.z)
            l_leg = math.dist(lheel_pos, (lhip.x, lhip.y, lhip.z))
            r_leg = math.dist(rheel_pos, (rhip.x, rhip.y, rhip.z))
            planted_leg = "left" if l_leg < r_leg else "right"

            left_stationary = right_stationary = False
            if prev_lheel:
                left_stationary = math.dist(lheel_pos, prev_lheel) < stationary_threshold_m
                right_stationary = math.dist(rheel_pos, prev_rheel) < stationary_threshold_m

            prev_lheel = lheel_pos
            prev_rheel = rheel_pos

            # Drawing
            cv2.circle(frame, (lx, ly), 8 if left_stationary else 6, (0, 0, 255), -1)
            cv2.circle(frame, (rx, ry), 8 if right_stationary else 6, (0, 0, 255), -1)
            cv2.circle(frame, (lhx, lhy), 6, (255, 0, 0), -1)
            cv2.circle(frame, (rhx, rhy), 6, (255, 0, 0), -1)
            cv2.line(frame, (lhx, lhy), (lx, ly), (255, 0, 0), 2)
            cv2.line(frame, (rhx, rhy), (rx, ry), (255, 0, 0), 2)

            # Distance in meters and feet
            stride_m = math.dist(lheel_pos, rheel_pos)
            stride_ft = stride_m * 3.28084

            # PRINT
            print(f"🦶 Frame {start_frame + frame_idx} | Time: {timestamp_str}")
            print(f"Left Heel:  ({lheel.x:.3f}, {lheel.y:.3f}, {lheel.z:.3f}) - {'STILL' if left_stationary else 'MOVING'}")
            print(f"Right Heel: ({rheel.x:.3f}, {rheel.y:.3f}, {rheel.z:.3f}) - {'STILL' if right_stationary else 'MOVING'}")
            print(f"👣 Stride: {stride_m:.3f} m ({stride_ft:.2f} ft) | Planted Leg: {planted_leg}")
            if left_stationary != right_stationary:
                print(f"✔️ Stationary Heel: {'left' if left_stationary else 'right'}")
            print()

            # === SAVE TO CSV ===
            with open(output_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    start_frame + frame_idx,
                    timestamp_str,
                    f"{stride_m:.3f}",
                    f"{stride_ft:.2f}",
                    planted_leg,
                    "STILL" if left_stationary else "MOVING",
                    "STILL" if right_stationary else "MOVING"
                ])

            if left_stationary != right_stationary:
                with open(stride_event_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        start_frame + frame_idx,
                        timestamp_str,
                        f"{stride_m:.3f}",
                        f"{stride_ft:.2f}",
                        planted_leg,
                        "left" if left_stationary else "right"
                    ])

        cv2.imshow("Stride Detection (WorldLandmarks)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

video.release()
cv2.destroyAllWindows()
