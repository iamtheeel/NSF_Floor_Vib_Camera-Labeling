## Imports
#Built ins
import time

#Third party
import cv2 # opencv-python
import pytesseract # pytesseract
import matplotlib as plt # matplotlib
import numpy as np # numpy

# Media Pipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_lite.task" # 5.5 MiB
model_path = r"C:\Users\smitt\STARS\pose_landmarker_full.task" # 9.0 MiB
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_heavy.task" # 29.2 MiB


#Video File
dir = r"E:\STARS\day1_data"
file = r"25_06_03_s1_1.asf"
fileName = f"{dir}/{file}"

#Global variables
videoOpbject = cv2.VideoCapture(fileName) #open the video file and make a video object
if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

#mediaPipe settings
### From https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#video ###
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
                                base_options=BaseOptions(model_asset_path=model_path,
                                                         delegate=BaseOptions.Delegate.CPU # Default is GPU, and I anin't got none
                                                         ),
                                running_mode=VisionRunningMode.VIDEO,
                               )

landmarker = PoseLandmarker.create_from_options(options)
#exit()

# Get video properties    
fps = videoOpbject.get(cv2.CAP_PROP_FPS) # Frames per second
#print(f"FPS: {fps}")
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT) #Frame count
height, width, _ = videoOpbject.read()[1].shape # Get the width and height of the video frame
#width = 256
#height = 400 

frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
#dispFact = 2
#displayRez = (int(width/dispFact), int(height/dispFact))

#functions

def drawLandmark_circle(frame, landmark):
    radius = 15
    thickness = 5
    color = [255, 0, 0] #Circle will be red
    center = int(landmark.x*width), int(landmark.y*height) #place center of the circle at the landmark position
    #print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")q
    cv2.circle(frame, center, radius, color, thickness) # Draw a circle at the landmark position

def drawLandmark_line(frame, feet, hips):
    color = [255, 0, 0] # Line will be red
    pt1_ft = (int(feet.x*width),int(feet.y*height)) #First point is on the feet
    pt2_hips = (int(hips.x*width), int(hips.y*height)) #second point is on the hips
    thickness = 5
    cv2.line(frame,pt1_ft,pt2_hips, color, thickness) # Draw a line from the feet to the hips

clipStartFrame = (fps *35) # Start frame for the clip

frame_timestamp_ms = clipStartFrame * frameTime_ms # Timestamp for the first frame in the clip

videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame)) # Initial start point
remainingFrames = int(fCount - clipStartFrame)
for i in range(int(remainingFrames)): # Go through each frame
    success, frame = videoOpbject.read()
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms)
    if not success:
        print("Failed to read frame")
        break
    
    newFrame = frame[0:250, 1188:1444]
    newFrame_rgb = cv2.cvtColor(newFrame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=newFrame_rgb) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms)) # Detect the pose landmarks in each frame
    """
    if len(pose_landmarker_result.pose_world_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        drawLandmark_circle(newFrame_rgb, landmarks[29]) # Draw circle on the left heel
        drawLandmark_line(newFrame_rgb, landmarks[29],landmarks[23]) # Draws line from left foot to left hip
        drawLandmark_line(newFrame_rgb, landmarks[30], landmarks[24]) # Draws line from right foot to right hip
        drawLandmark_circle(newFrame_rgb, landmarks[30]) # Draw circle on the right heel
        """
    if pose_landmarker_result.pose_landmarks:
        for i, landmark in enumerate(pose_landmarker_result.pose_landmarks[0]):
            if landmark.visibility > 0.5:  # Or even 0.3 for partials
                print(f"Landmark {i} detected at ({landmark.x:.2f}, {landmark.y:.2f})")
    else:
        print(f"No pose detected at frame {i}, time {frame_timestamp_ms} ms")
    cv2.imshow("Frame", newFrame_rgb)
    
    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()