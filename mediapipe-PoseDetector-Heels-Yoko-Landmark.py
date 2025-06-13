####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# PoseLandmarks (2D only) — Heel Stride and X-Position in Meters, One-Line Output
####

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pytesseract
import csv

# === CALIBRATION ===
cm_per_pixel = 1 / 3.0
m_per_pixel = cm_per_pixel / 100.0
ft_per_pixel = cm_per_pixel / 30.48

# === VIDEO & MODEL ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

video = cv2.VideoCapture(video_file)
if not video.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
frameTime_ms = 1000 / fps
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === MEDIAPIPE LANDMARKER SETUP ===
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

# === OCR TIMESTAMP ===
def getDateTime(frame):
    roi = frame[0:45, 0:384]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    data = pytesseract.image_to_data(inverted, output_type=pytesseract.Output.DICT)
    try:
        return f"{data['text'][4]} {data['text'][5]}"
    except:
        return "Unknown Time"

# === CSV SETUP ===
output_csv = "stride_data_meters.csv"
event_csv = "stride_events_meters.csv"
with open(output_csv, 'w', newline='') as f:
    csv.writer(f).writerow(["Frame", "Timestamp", "Stride_m", "Stride_ft", "Planted_Leg", "LeftHeelX_m", "RightHeelX_m", "LeftHeelStatus", "RightHeelStatus"])
with open(event_csv, 'w', newline='') as f:
    csv.writer(f).writerow(["Frame", "Timestamp", "Stride_m", "Stride_ft", "Planted_Leg", "Stationary_Heel"])

# === TRACKING VARS ===
prev_lx, prev_ly = None, None
prev_rx, prev_ry = None, None
stationary_threshold_px = 10

# === MAIN LOOP ===
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

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Get pixel coordinates
            def to_px(lm): return int(lm.x * w), int(lm.y * h)
            lx, ly = to_px(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value])
            rx, ry = to_px(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value])
            lhx, lhy = to_px(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value])
            rhx, rhy = to_px(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value])

            # Leg lengths
            l_leg = math.hypot(lx - lhx, ly - lhy)
            r_leg = math.hypot(rx - rhx, ry - rhy)
            planted_leg = "left" if l_leg < r_leg else "right"

            # Heel motion detection
            left_stationary = right_stationary = False
            if prev_lx is not None:
                left_stationary = abs(lx - prev_lx) < stationary_threshold_px and abs(ly - prev_ly) < stationary_threshold_px
                right_stationary = abs(rx - prev_rx) < stationary_threshold_px and abs(ry - prev_ry) < stationary_threshold_px

            prev_lx, prev_ly = lx, ly
            prev_rx, prev_ry = rx, ry

            # Stride
            stride_px = math.hypot(lx - rx, ly - ry)
            stride_m = stride_px * m_per_pixel
            stride_ft = stride_px * ft_per_pixel

            # Heel x-positions in meters (horizontal position only)
            lx_m = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value].x * w * m_per_pixel
            rx_m = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value].x * w * m_per_pixel

            # === One-line output ===
            output_line = (
                f"🦶 Frame {start_frame + frame_idx} | {timestamp_str} | "
                f"Stride: {stride_m:.3f} m ({stride_ft:.2f} ft) | "
                f"Planted: {planted_leg} | "
                f"Left Heel X: {lx_m:.3f} m - {'STILL' if left_stationary else 'MOVING'} | "
                f"Right Heel X: {rx_m:.3f} m - {'STILL' if right_stationary else 'MOVING'}"
            )
            if left_stationary != right_stationary:
                output_line += f" | Stationary Heel: {'left' if left_stationary else 'right'}"
            print(output_line)

            # === CSV LOGGING ===
            with open(output_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    start_frame + frame_idx,
                    timestamp_str,
                    f"{stride_m:.3f}",
                    f"{stride_ft:.2f}",
                    planted_leg,
                    f"{lx_m:.3f}",
                    f"{rx_m:.3f}",
                    "STILL" if left_stationary else "MOVING",
                    "STILL" if right_stationary else "MOVING"
                ])
            if left_stationary != right_stationary:
                with open(event_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        start_frame + frame_idx,
                        timestamp_str,
                        f"{stride_m:.3f}",
                        f"{stride_ft:.2f}",
                        planted_leg,
                        "left" if left_stationary else "right"
                    ])

        cv2.imshow("Stride Detection (Meters)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

video.release()
cv2.destroyAllWindows()
