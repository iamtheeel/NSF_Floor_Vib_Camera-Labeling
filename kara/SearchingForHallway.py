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
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_full.task" # 9.0 MiB
model_path = r"C:\Users\smitt\STARS\pose_landmarker_heavy.task" # 29.2 MiB


#Video File
dir = r"E:\STARS\day1_data"
file = r"25_06_03_s1_1.asf"
fileName = f"{dir}/{file}"

#Global variables
videoOpbject = cv2.VideoCapture(fileName) #open the video file and make a video object
if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties    
fps = videoOpbject.get(cv2.CAP_PROP_FPS) # Frames per second
#print(f"FPS: {fps}")
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT) #Frame count
height, width, _ = videoOpbject.read()[1].shape # Get the width and height of the video frame
#width = 256
#height = 256 

frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
dispFact = 2
displayRez = (int(width/dispFact), int(height/dispFact))

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

landmarkerVideo = PoseLandmarker.create_from_options(options)
#exit()

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

def isPersonInFrame(frame_Index):
    
    videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, frame_Index) # Set the video object to the frame we want to check
    
    ret, frame = videoOpbject.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_timestamp_ms = int(frame_Index * frameTime_ms)

    if frame_timestamp_ms < 0 or frame_timestamp_ms > 1e10:
        print(f"Invalid timestamp: {frame_timestamp_ms}")
        return False

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_RGB) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarkerVideo.detect_for_video(mp_image, frame_timestamp_ms)
    # Detect the pose landmarks in each frame
    
    if len(pose_landmarker_result.pose_landmarks) > 0:
        print(f"There is a person!") # If there is a pose landmarker, return True
    else:
        print(f"No person detected at {frame_timestamp_ms} ms")
    #return False # If there is no pose landmarker, return False
    return frame

def crop_with_padding(frame, lefttHip, rightHip, rightFoot, crop_width=256):
    # Calculate center between left and right hips
    hip_x = (rightHip + lefttHip) // 2 # Calculate the center x position between the left and right hips
    # Set crop bounds equidistant from the center
    x1 = hip_x - crop_width // 2
    x2 = hip_x + crop_width // 2
    # Calculate how much we are out of bounds
    pad_left   = max(0, -x1) #x1 is negative if it extends beyond the left edge of the frame
    pad_right  = max(0, x2 - width) #x1 is negative if it extends beyond the right edge of the frame
        #Take max of numbers to get padding amount
    # Apply padding if needed
    frame_padded = cv2.copyMakeBorder(frame,0,0,pad_left,pad_right,borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))  # Black padding
     # Update crop coordinates for the padded image
    x1_padded = x1 + pad_left
    x2_padded = x2 + pad_right
    #y1_padded = y1 + pad_top
    #y2_padded = y2 + pad_top
    # Crop the padded frame
    cropped = frame_padded[0: rightFoot + 50, x1_padded:x2_padded]
    print(f"Crop bounds: x1: {x1_padded}, x2: {x2_padded}, y1: 0, y2: {rightFoot + 50}")
    return cropped   #Return the cropped frame with padding

# Main code

frame_Index = 50 # Frame index to check for a person

frame = isPersonInFrame(frame_Index) # Check if there is a person in the frame

if frame is not None:
    ret, frame = videoOpbject.read()
    if not ret:
        print("Error: Could not read frame.")
    resizedFrame = cv2.resize(frame, displayRez) # Resize the frame for display
    cv2.imshow("Frame", resizedFrame)
    key = cv2.waitKey(0) # Wait for a key press
    if key == ord('q') & 0xFF: exit()
    
else:
    print("No frame returned, exiting.")
    exit()




#if isPersonInFrame(frame_Index):
   # print("Person detected in the first frame.")
#else:
    #print("No person detected in the first frame.")



"""
startTime = 30

clipStartFrame = (fps * startTime) # Start frame for the clip

#frame_timestamp_ms = clipStartFrame * frameTime_ms # Timestamp for the first frame in the clip

#videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame)) # Initial start point

remainingFrames = int(fCount - clipStartFrame)

for i in range(int(remainingFrames)): # Go through each frame
    success, frame = videoOpbject.read() # Read the next frame returns a boolean and the frame
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms) # Update the timestamp for the current frame
    if not success: # If the frame was not read successfully, break the loop
        print("Failed to read frame")
        break
    
    #frame = frame[0:400, 1000:1444, :] # Crop the frame to the area of interest
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms)) # Detect the pose landmarks in each frame
    
    if len(pose_landmarker_result.pose_world_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        drawLandmark_circle(frame, landmarks[29]) # Draw circle on the left heel
        drawLandmark_line(frame, landmarks[29],landmarks[23]) # Draws line from left foot to left hip
        drawLandmark_line(frame, landmarks[30], landmarks[24]) # Draws line from right foot to right hip
        drawLandmark_circle(frame, landmarks[30]) # Draw circle on the right heel
       
    if pose_landmarker_result.pose_landmarks:
        for i, landmark in enumerate(pose_landmarker_result.pose_landmarks[0]):
            if landmark.visibility > 0.5:  # Or even 0.3 for partials
                print(f"Landmark {i} detected at ({landmark.x:.2f}, {landmark.y:.2f})")
       
    else:
        print(f"No pose detected at frame {i}, time {frame_timestamp_ms} ms")
    resizedFrame = cv2.resize(frame, displayRez) # Resize the frame for display
    cv2.imshow("Frame", resizedFrame)
    
    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()
 """