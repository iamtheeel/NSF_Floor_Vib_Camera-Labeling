####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking with OCR Timestamp 
####

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import pytesseract  # Needed for getDateTime()

### From https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/#video
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task" #9.4MB
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task" #30.7MB
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task" #5.8B

# === SETUP ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"
video_file = "/Volumes/MY PASSPORT/Stars_day1Data/s2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf"

video = cv2.VideoCapture(video_file)
if not video.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)q
frameTime_ms = 1000 / fps
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === POSE DETECTOR SETUP ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#What model file to load and what device to use
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.VIDEO,
)

# === FRAME RANGE ===
startSecs = 125
endSecs = 155
start_frame = int(fps * startSecs)
end_frame = int(fps * endSecs)
num_frames = end_frame - start_frame

if num_frames < 0:
    print("⚠️ Invalid frame range.")
    exit()

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# === OCR TIMESTAMP FUNCTION ===
# OCR = Optical Character Recognition. Read and extract text from images or video frames.
def getDateTime(frame):
    dateTime_img = frame[0:45, 0:384]
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)
    dateTime_img_bw = cv2.bitwise_not(dateTime_img_bw)
    data = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    try:
        date_str = data['text'][4] #Get 5th detected word that is from OCR result and assume it is date.
        time_str = data['text'][5]
        conf = int(data['conf'][4])
        timestamp = f"{date_str} {time_str}"
        return timestamp
    except:
        return "Unknown Time"
    
# === CALIBRATION: 1 cm = 3 pixels (adjust based on known marker)
# Convert pixel distances from the video into real-world physical units.
# Adjust number. Assuming --> the distance between heels is 1 cm apart.
cm_per_pixel = 1 / 3.0

# === POSE PROCESSING ===
with PoseLandmarker.create_from_options(options) as detector:
    frame_idx = 0 #Starts the frame counter at zero, tracks how many frames have been processed.
    while frame_idx < num_frames: #Keeps looping until you've processed the desired number of frames (from start_frame to end_frame).
        success, frame = video.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) #Wraps the RGB frame into a MediaPipe Image object. SRGB is pixel.
        frame_timestamp_ms = int((start_frame + frame_idx) * frameTime_ms) #Calculates the timestamp (in milliseconds) for the current frame.

        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        timestamp_str = getDateTime(frame)  # OCR timestamp

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0] #Gets the list of landmarks (33 in total) for the first detected person. X, Y, Z normalized and Visibility score.
            left_heel = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value]
            right_heel = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value]

            lx, ly = int(left_heel.x * w), int(left_heel.y * h) #28 code for pose landmarker left heel.
            rx, ry = int(right_heel.x * w), int(right_heel.y * h) #29 - same as above, right_heel.x and .y are normalized.

            # Draw red dots
            cv2.circle(frame, (lx, ly), 6, (0, 0, 255), -1) #l = left, 6 = radius, -1 = fill the circle (solid dot)
            cv2.circle(frame, (rx, ry), 6, (0, 0, 255), -1)

            # Calculate stride distance in cm and ft
            heel_px = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
            heel_cm = heel_px * cm_per_pixel
            heel_ft = heel_cm / 30.48

            # Final print with cm and ft
            print(f"{timestamp_str} | Frame {start_frame + frame_idx} | Stride: {heel_cm:.1f} cm ({heel_ft:.2f} ft)")

        cv2.imshow("Stride Detection (Left and Right Heels)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

video.release()
cv2.destroyAllWindows()
