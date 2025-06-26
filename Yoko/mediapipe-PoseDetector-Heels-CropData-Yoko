# === Full Working Script: Pose Tracking with MediaPipe and OCR + CSV Output ===

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
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"

# === VIDEO FILE ===
video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/25_06_18/Subject_1'
video_file = 'Sub_1_Run_1_6-18-2025_11-45-46 AM.asf'
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
width = w
height = h

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

def drawLandmark_circle(frame, landmark, color):
    radius = 15
    thickness = 5
    center = int(landmark.x*width), int(landmark.y*height)
    cv2.circle(frame, center, radius, color, thickness)

def drawLandmark_line(frame, feet, hips, color):
    pt1_ft = (int(feet.x*width),int(feet.y*height))
    pt2_hips = (int(hips.x*width), int(hips.y*height))
    thickness = 5
    cv2.line(frame,pt1_ft,pt2_hips, color, thickness)

def drawLandmark_square(frame, minWidth, maxWidth, minHeight, maxHeight):
    color = [255, 0, 0]
    xyPt = int(minWidth),int(minHeight)
    XyPt = int(maxWidth), int(minHeight)
    XYPt = int(maxWidth), int(maxHeight)
    xYPt = int(minWidth), int(maxHeight)
    thickness = 5
    cv2.line(frame, xyPt, XyPt, color, thickness) 
    cv2.line(frame, XyPt, XYPt, color, thickness)
    cv2.line(frame, XYPt, xYPt, color, thickness)
    cv2.line(frame, xYPt, xyPt, color, thickness)

def crop_to_ratio(frame, landmarks):
    frame_height, frame_width = frame.shape[:2]
    core_landmarks = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
    min_width = max_width = landmarks[0].x
    min_height = max_height = landmarks[0].y

    for i in core_landmarks:
        x = landmarks[i].x
        y = landmarks[i].y
        min_width = min(min_width, x)
        max_width = max(max_width, x)
        min_height = min(min_height, y)
        max_height = max(max_height, y)

    min_width *= frame_width
    max_width *= frame_width
    min_height *= frame_height
    max_height *= frame_height

    tot_width = max_width - min_width
    tot_height = max_height - min_height
    Ratio = width/height
    current_ratio = tot_width / tot_height
    center_width = min_width + tot_width / 2
    center_height = min_height + tot_height / 2

    if current_ratio < Ratio:
        adjust_width = (tot_height * Ratio) / 2
        min_width = center_width - adjust_width
        max_width = center_width + adjust_width
    else:
        adjust_height = (tot_width / Ratio) / 2
        min_height = center_height - adjust_height
        max_height = center_height + adjust_height

    tot_width = max_width - min_width
    tot_height = max_height - min_height
    center_width = min_width + tot_width / 2
    center_height = min_height + tot_height / 2

    scale_factor = 1.9

    new_width = tot_width * scale_factor
    new_height = tot_height * scale_factor
    
    min_width = center_width - new_width / 2
    max_width = center_width + new_width / 2
    min_height = center_height - new_height / 2
    max_height = center_height + new_height / 2

    min_width = max(0, round(min_width))
    max_width = min(frame_width, round(max_width))
    min_height = max(0, round(min_height))
    max_height = min(frame_height, round(max_height))

    if max_width <= min_width or max_height <= min_height:
        return 0, frame_width, 0, frame_height

    return round(min_width), round(max_width), round(min_height), round(max_height)

def crop_to_square(frame, landmarks):
    #Checks if there are landmarkers 

    frame_height, frame_width = frame.shape[:2] 

    # Use only major body parts that are symmetrical and close to the torso
    core_landmarks = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]  # shoulders, hips, knees, etc.

    min_width = max_width = landmarks[0].x #Initiates width variables to landmark 0
    min_height = max_height = landmarks[0].y #Initiates height variables to landmark 0

    #Iterates through all landmarks to find max and min: x value = width, y value = height
    for i in core_landmarks:
        x = landmarks[i].x
        y = landmarks[i].y
        if x < min_width:
            min_width = x
        if x > max_width:
            max_width = x
        if y < min_height:
            min_height = y
        if y > max_height:
            max_height = y
    
    #Normalize values to frame
    min_width = min_width*frame_width
    max_width = max_width*frame_width
    min_height=min_height*frame_height
    max_height=max_height*frame_height

    tot_width = max_width - min_width   #total width of cropped frame
    tot_height = max_height - min_height #total height of cropped frame
    
    #Finds the center WRT full frame by adding half of width/height of 
    #cropped screen to min height/width in full frame dimensions
    center_width = min_width + tot_width / 2 
    center_height = min_height + tot_height / 2
    
    adjust_width = tot_height//2
    min_width = center_width - adjust_width
    max_width = center_width + adjust_width
    tot_width = max_width - min_width   #total width of cropped frame

    scale_factor = 1.9

    #scales total width/height
    new_width = tot_width * scale_factor
    new_height = tot_height * scale_factor
    
    #Calculates new min/max height/width WRT to full frame by adding 
    #center (in full frame coords) to width/height (in cropped frame coords)
    min_width = center_width - new_width / 2
    max_width = center_width + new_width / 2
    min_height = center_height - new_height / 2
    max_height = center_height + new_height / 2

    #Ensures that crop is within bounds (0 to full frame size)
    min_width = max(0, round(min_width))
    max_width = min(frame_width, round(max_width))
    min_height = max(0, round(min_height))
    max_height = min(frame_height, round(max_height))

    # Make sure the result isn't an empty crop
    if max_width <= min_width or max_height <= min_height:
    # Return the full frame as fallback
        return 0, frame_width, 0, frame_height
    
    return round(min_width), round(max_width), round(min_height), round(max_height)

# === CSV SETUP ===
csv_path = "heel_tracking_output.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Frame", "Timestamp_ms",
        "LeftHeel_Y", "RightHeel_Y",
        "LeftHeel_Distance", "RightHeel_Distance",
        "LeftHeel_Visibility", "RightHeel_Visibility",
        "LeftHeel_Presence", "RightHeel_Presence"
    ])

# === Frame Timing ===
frameTime_ms = 1000 / fps
clipStartTime_s = 59
clipRunTime_s = 0
clipStartFrame = int(clipStartTime_s * fps)
clipRunFrames = int((fCount - clipStartFrame) if clipRunTime_s == 0 else clipRunTime_s * fps)
videoObject.set(cv2.CAP_PROP_POS_FRAMES, clipStartFrame)

# === Main Frame Loop ===
for i in range(clipRunFrames):
    success, frame = videoObject.read()
    if not success:
        print("⚠️ Frame read failure")
        break

    frame_timestamp_ms = i * frameTime_ms
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))

    if len(pose_landmarker_result.pose_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        landmarks_w = pose_landmarker_result.pose_world_landmarks[0]

        #Change the code for fitting the crop
        min_width, max_width, min_height, max_height = crop_to_ratio(frame, landmarks) 
        #min_width, max_width, min_height, max_height = crop_to_square(frame, landmarks)

        drawLandmark_square(frame, min_width, max_width, min_height, max_height)
        drawLandmark_circle(frame, landmarks[29], [255, 0, 0]) #left heel 
        drawLandmark_circle(frame, landmarks[30], [0, 0, 255]) #right heel
        drawLandmark_line(frame, landmarks[29], landmarks[23], (255, 0, 0))
        drawLandmark_line(frame, landmarks[30], landmarks[24], (0, 0, 255))

        # === Use find_dist_from_y for both heels ===
        # Get normalized Y (0.0 to 1.0) for print
        left_heel = landmarks[29]
        right_heel = landmarks[30]

        # Convert to pixel Y for distance calculation
        left_heel_y_norm = left_heel.y
        right_heel_y_norm = right_heel.y
        left_heel_y_px = left_heel_y_norm * h
        right_heel_y_px = right_heel_y_norm * h

        # Compute distance
        left_dist = find_dist_from_y(left_heel_y_px, debug=True)
        right_dist = find_dist_from_y(right_heel_y_px, debug=True)

        print(f"Left Heel - visibility: {left_heel.visibility:.4f}, presence: {left_heel.presence:.4f}")
        print(f"Right Heel - visibility: {right_heel.visibility:.4f}, presence: {right_heel.presence:.4f}")

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                i,
                frame_timestamp_ms,
                left_heel_y_norm,
                right_heel_y_norm,
                left_dist,
                right_dist,
                left_heel.visibility,
                right_heel.visibility,
                left_heel.presence,
                right_heel.presence
            ])

    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoObject.release()
cv2.destroyAllWindows()

