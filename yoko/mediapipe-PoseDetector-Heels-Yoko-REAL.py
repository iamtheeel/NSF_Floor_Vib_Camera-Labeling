####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# Playing with media pipe
# Open our video file
####

## Imports
# Built-ins
import time
import math

# Third party
import cv2  # pip install opencv-python
import numpy as np
import pytesseract  # pip install pytesseract to do OCR/ICR

# Media Pipe
import mediapipe as mp  # pip install mediapipe
from mediapipe.tasks import python  # installs with mediapipe
from mediapipe.tasks.python import vision  # installs with mediapipe

# === MODEL PATH ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task"  # 9.4MB
# model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"  # 30.7MB
# model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task"  # 5.8MB

# === VIDEO FILE ===
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

# Open video file
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = videoObject.get(cv2.CAP_PROP_FPS)
fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT)
w = videoObject.get(cv2.CAP_PROP_FRAME_WIDTH)
h = videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT)

frameTime_ms = 1000 / fps  # Time each frame covers in milliseconds

dispFact = 2 #shrink the image to half size
displayRez = (int(w / dispFact), int(h / dispFact))

# MediaPipe PoseLandmarker setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path,
        delegate=BaseOptions.Delegate.CPU  # Use CPU (default is GPU)
    ),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=True
)
landmarker = PoseLandmarker.create_from_options(options)

# === OCR Timestamp Setup ===
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
    radius = 6
    thickness = -1
    center = (int(landmark.x * w), int(landmark.y * h)) #x and y are normalized coordinates between the value of 0 and 1
    cv2.circle(frame, center, radius, color, thickness)

def drawLine(frame, lm1, lm2, color):
    pt1 = (int(lm1.x * w), int(lm1.y * h)) # converts landmarks lm1 and lm2 to pixel positions
    pt2 = (int(lm2.x * w), int(lm2.y * h))
    cv2.line(frame, pt1, pt2, color, 3) #draw colored line, 3 = thickness

def calc_dist(p1, p2): #calculates the 3D Euclidean distance between two points (in meters if using pose_world_landmarks)
    return math.sqrt(
        (p1.x - p2.x)**2 + #These three lines are inside the square root, gives squared differences between the coordinates
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2
    )

# Clip setup
clipRunTime_s = 0
clipStartTime_s = 30

clipEndTime_s = clipStartTime_s + clipRunTime_s
clipStartFrame = clipStartTime_s * fps
clipRunFrames = int((fCount - clipStartFrame) if clipRunTime_s == 0 else (clipEndTime_s - clipStartTime_s) * fps)

videoObject.set(cv2.CAP_PROP_POS_MSEC, clipStartTime_s * 1000) #MSEC = Milliseconds

# Frame processing loop
frame_timestamp_ms = 0
for i in range(clipRunFrames):
    frame_timestamp_ms += int(i * frameTime_ms)
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

        # Draw 2D landmarks
        drawLandmark(frame, landmarks[29], [255, 0, 0])    # Left heel
        drawLandmark(frame, landmarks[30], [0, 255, 0])    # Right heel
        drawLandmark(frame, landmarks[23], [255, 255, 0])  # Left hip
        drawLandmark(frame, landmarks[24], [0, 255, 255])  # Right hip

        drawLine(frame, landmarks[29], landmarks[23], (255, 100, 100))
        drawLine(frame, landmarks[30], landmarks[24], (100, 255, 100))

        # 3D distances (meters)
        strideLen = calc_dist(landmarks_w[29], landmarks_w[30])
        left_leg_len = calc_dist(landmarks_w[23], landmarks_w[29])
        right_leg_len = calc_dist(landmarks_w[24], landmarks_w[30])

        #print(f"📹 Frame {i}, Timestamp: {frame_timestamp_ms} ms | Stride: {strideLen:.3f} m | {strideLen * 3.28084:.2f} ft | Left Leg: {left_leg_len:.3f} m | {left_leg_len * 3.28084:.2f} ft | Right Leg: {right_leg_len:.3f} m | {right_leg_len * 3.28084:.2f} ft")
        print(f"Timestamp: {frame_timestamp_ms} ms | left heel: {landmarks[29].y}, right heel: {landmarks[30].y}")  

        # Segmentation mask
        # If MediaPipe Pose was able to separate the person from the background, then show that separation (called a segmentation mask) as a grayscale image.
        if pose_landmarker_result.segmentation_masks is not None: #A special output from MediaPipe that says which part of the video is the person’s body, and which part is the background.
            mask = pose_landmarker_result.segmentation_masks[0].numpy_view() #Converts MediaPipe’s internal data into something we can work with using Python (a NumPy array).
            cv2.imshow("Seg mask", mask)

    # Show annotated frame
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)

    key = cv2.waitKey(1)
    if key == ord('q') & 0xFF:
        break

videoObject.release()
cv2.destroyAllWindows()
