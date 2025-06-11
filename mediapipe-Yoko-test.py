####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking with OCR Timestamp (Clean Output)
# Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
####

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pytesseract
import numpy as np
import math

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# === MODEL PATH ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"

# === CONFIGURATION ===
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

# === OPEN VIDEO ===
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Get properties from the video
fps = videoObject.get(cv2.CAP_PROP_FPS)
fCount = int(videoObject.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameTime_ms = 1000 / fps
dispFact = 2
displayRez = (int(w) // dispFact, int(h) // dispFact)

# === MEDIAPIPE SETUP ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

# === OCR FUNCTION ===
def getDateTime(frame):
    crop = frame[0:45, 0:384]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    data = pytesseract.image_to_data(inv, output_type=pytesseract.Output.DICT)
    try:
        date_str = data['text'][4]
        time_str = data['text'][5]
        conf = int(data['conf'][4])
        return f"{date_str} {time_str}", conf
    except:
        return "OCR Error", 0

# === LANDMARK LOGGING FUNCTION ===
def print_landmarks(result):
    if not result.pose_landmarks:
        return
    for idx, landmark in enumerate(result.pose_landmarks[0]):
        x = landmark.x
        y = landmark.y
        z = landmark.z
        visibility = landmark.visibility
        print(f"Landmark {idx}: x={x:.3f}, y={y:.3f}, z={z:.3f}, visibility={visibility:.2f}")

# === FRAME RANGE ===
startSecs = 125
endSecs = 155
start_frame = int(fps * startSecs)
end_frame = int(fps * endSecs)
num_frames = end_frame - start_frame

if num_frames < 0:
    print(f"⚠️ Invalid frame range.")
    exit()

videoObject.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# === MAIN DETECTION LOOP ===
with PoseLandmarker.create_from_options(options) as landmarker:
    for i in range(num_frames):
        frame_timestamp_ms = int((start_frame + i) * frameTime_ms)
        success, frame = videoObject.read()

        if not success:
            print("❌ Frame read failed.")
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect pose
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        print(f"🔍 Frame {i+1}/{num_frames} - Landmarks detected: {bool(result.pose_landmarks)}")

        # Print landmark info
        print_landmarks(result)

        # OCR timestamp from frame
        timestamp_text, conf = getDateTime(frame)
        print(f"📅 Date & Time: {timestamp_text}, Confidence: {conf}")

        # Resize and show frame
        frame_small = cv2.resize(frame, displayRez)
        cv2.imshow("📹 Video Frame Only", frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔴 Quit by user.")
            break

# === CLEANUP ===
videoObject.release()
cv2.destroyAllWindows()
