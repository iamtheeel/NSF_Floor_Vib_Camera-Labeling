## Imports
#Built ins
import time

#Third party
import cv2 # opencv-python
import pytesseract # pytesseract
import matplotlib as plt # matplotlib
import numpy as np # numpy
import csv

# Media Pipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys 
import os

# === Fix import path to reach distance_position.py ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_position import find_dist_from_y  # ✅ Import your custom function

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
#height, width, _ = videoOpbject.read()[1].shape doing this reads the first frame, which we don't want to do yet
width = int(videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH)) # Width of the video frame
height = int(videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of the video frame
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
                                #running_mode=VisionRunningMode.VIDEO,
                                running_mode=VisionRunningMode.VIDEO,
                               )

landmarkerVideo = PoseLandmarker.create_from_options(options)
#exit()

#functions

def drawLandmark_circle(frame, landmark, color):
    radius = 15
    thickness = 5
    #color = [255, 0, 0] #Circle will be red
    #center = int(center_width), int(center_height)
    center = int(landmark.x*width), int(landmark.y*height) #place center of the circle at the landmark position
    #print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")q
    cv2.circle(frame, center, radius, color, thickness) # Draw a circle at the landmark position

def drawLandmark_line(frame, feet, hips, color):
    #color = [255, 0, 0] # Line will be red
    pt1_ft = (int(feet.x*width),int(feet.y*height)) #First point is on the feet
    pt2_hips = (int(hips.x*width), int(hips.y*height)) #second point is on the hips
    thickness = 5
    cv2.line(frame,pt1_ft,pt2_hips, color, thickness) # Draw a line from the feet to the hips
    
def drawLandmark_square(frame, minWidth, maxWidth, minHeight, maxHeight):
    color = [255, 0, 0] # Line will be red
    #pt1_ft = (int(feet.x*width),int(feet.y*height)) #First point is on the feet
    #pt2_hips = (int(hips.x*width), int(hips.y*height)) #second point is on the hips
    xyPt = int(minWidth),int(minHeight) #upper left pt
    XyPt = int(maxWidth), int(minHeight) #upper right pt
    XYPt = int(maxWidth), int(maxHeight) #lower right pt
    xYPt = int(minWidth), int(maxHeight) #lower left pt
    thickness = 5
    #Connects points to draw a square
    cv2.line(frame, xyPt, XyPt, color, thickness) 
    cv2.line(frame, XyPt, XYPt, color, thickness) #
    cv2.line(frame, XYPt, xYPt, color, thickness)
    cv2.line(frame, xYPt, xyPt, color, thickness)

def isPersonInFrame(frame, frameIndex):

    frame_timestamp_ms = int(frameIndex * frameTime_ms) 
    if frame_timestamp_ms < 0 or frame_timestamp_ms > 1e10: # Check if the timestamp is valid
        #print(f"Invalid timestamp: {frame_timestamp_ms}")
        return None #Exit function if the timestamp is invalid

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarkerVideo.detect_for_video(mp_image, frame_timestamp_ms) #Detect the pose landmarks in the frame
    
    #If there are no pose landmarkers
    if len(pose_landmarker_result.pose_landmarks) > 0: 
        return True, pose_landmarker_result, frame_timestamp_ms
    else:
        return False, None, frame_timestamp_ms

def crop_with_padding(frame, landmarks):
    #Checks if there are landmarkers 
    
    #landmarks = pose_landmarker_result.pose_landmarks[0]

    frame_height, frame_width = frame.shape[:2] 

    # Use only major body parts that are symmetrical and close to the torso
    core_landmarks = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]  # shoulders, hips, knees, etc.

    min_width = max_width = landmarks[0].x #Initiates width variables to landmark 0
    min_height = max_height = landmarks[0].y #Initiates height variables to landmark 0
    min_width_index = max_width_index = min_height_index = max_height_index = 0

    #Iterates through all landmarks to find max and min: x value = width, y value = height
    for i in core_landmarks:
        x = landmarks[i].x
        y = landmarks[i].y
        if x < min_width:
            min_width = x
            min_width_index = i
        if x > max_width:
            max_width = x
            max_width_index = i
        if y < min_height:
            min_height = y
            min_height_index = i
        if y > max_height:
            max_height = y
            max_height_index = i
    
    #Normalize values to frame
    min_width = min_width*frame_width
    max_width = max_width*frame_width
    min_height=min_height*frame_height
    max_height=max_height*frame_height

    tot_width = max_width - min_width   #total width of cropped frame
    tot_height = max_height - min_height #total height of cropped frame
    
    Ratio = width/height # Ratio of height/width of full sized frame

    current_ratio = tot_width / tot_height #Ratio of height/width of cropped frame
    
    #Finds the center WRT full frame by adding half of width/height of 
    #cropped screen to min height/width in full frame dimensions
    center_width = min_width + tot_width / 2 
    center_height = min_height + tot_height / 2
    
    #Change height/width ratio of cropped frame to match that of full frame
    if current_ratio < Ratio:
    # Too narrow: increase width (or crop height)
        adjust_width = (tot_height * Ratio) / 2
        min_width = center_width - adjust_width
        max_width = center_width + adjust_width
        #print(f"Width adjusted. Min width {min_width}. Max width {max_width}")
    else:
    # Too wide: crop width (or increase height)
        adjust_height = (tot_width / Ratio) / 2
        min_height = center_height - adjust_height
        max_height = center_height + adjust_height
        #print(f"Height adjusted. Min height {min_height}. Max height {max_height}")
    #adjusts total width/height according to new dimensions
    tot_width = max_width - min_width
    tot_height = max_height - min_height
    
    #adjusts center according to new dimensions
    center_width = min_width + tot_width / 2
    center_height = min_height + tot_height / 2
    
    scale_factor = 1.5

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


# Main code
start_frame = 0 # Start frame for the clip
end_frame = int(fCount) 
print("Initial frame position:", videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)) #Ensures initial frame is 0

# Read frames until we reach the frame prior to start frame
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

#initializes height/width to full size
max_height = height
min_height = 0
max_width = width
min_width = 0

# === CSV SETUP (✅ correct position: outside loop) ===
csv_path = r"E:\STARS\myfile2.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Frame", "Timestamp_ms",
        "LeftHeel_Y", "RightHeel_Y",
        "LeftHeel_Distance", "RightHeel_Distance"
    ])
#Opens 

#Read through the specified frame count
for frame_Index in range(start_frame, end_frame): 
    success, raw_frame = videoOpbject.read() # Returns a boolean and the next frame
    if not success: # If the frame was not read successfully, break the loop
        print("Failed to read frame")
        exit()
    newDim_Frame = raw_frame[min_height:max_height,min_width:max_width,:].copy() #Taking a full sized frame and 
    cv2.imshow("Frame to send model", newDim_Frame) #displays frame
    #Shrinking it down to dimensions
    #Changes dimensions before finding landmarks
    drawLandmark_square(raw_frame, min_width, max_width, min_height, max_height) #Returns a box around the person
    if newDim_Frame is not None: #Failsafe "if newDim_Frame is not None:"
    #good, result = isPersonInFrame(newDim_Frame, frame_Index) #newDim_Frame Checks if there is a person in the frame. Returns frame and landmarkers.
    #rescale and reshift
        good = False
        good, result, adjusted_time_ms = isPersonInFrame(newDim_Frame, frame_Index) #newDim_Frame Checks if there is a person
        #result is the landmarks
        if good and result is not None:
            landmarks = result.pose_landmarks[0]
            #drawLandmark_circle(raw_frame, landmarks[29], [255,0,0]) # Draw green landmarks before transition
            #drawLandmark_line(raw_frame, landmarks[29],landmarks[23], [255,0,0]) # Draw green landmarks before transition
            #drawLandmark_line(raw_frame, landmarks[30], landmarks[24], [255,0,0]) # Draw green landmarks before transition
            #drawLandmark_circle(raw_frame, landmarks[30], [255,0,0]) # Draw green landmarks before transition
            for i in range(len(landmarks)):   
                landmarks[i].x = (landmarks[i].x * (max_width - min_width) + min_width) / width
                landmarks[i].y = (landmarks[i].y * (max_height - min_height) + min_height) / height
            #drawLandmark_circle(raw_frame, landmarks[29], [0,255,0]) # Draw blue landmarks after transition
            #drawLandmark_line(raw_frame, landmarks[29],landmarks[23], [0,255,0]) # Draw blue landmarks after transition
            #drawLandmark_line(raw_frame, landmarks[30], landmarks[24], [0,255,0]) # Draw blue landmarks after transition
            #drawLandmark_circle(raw_frame, landmarks[30], [0,255,0]) # Draw blue landmarks after transition

            min_width, max_width, min_height, max_height = crop_with_padding(raw_frame, landmarks) #, landmarks
        #print(f"BACK IN MAIN for frame: {videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)} Minwidth: {min_width}. Maxwidth: {max_width} ")
        #new_Frame = crop_with_padding(raw_frame, landmarks) #Returns cropped frame
        #resizedFrame = cv2.resize(new_Frame, displayRez) # Resize the frame for display
        
        # === Heel Y values (normalized and pixel)
            left_heel_y_norm = landmarks[29].y
            right_heel_y_norm = landmarks[30].y
            left_heel_y_px = left_heel_y_norm * height
            right_heel_y_px = right_heel_y_norm * height
            # === Distances using your function
            left_dist = find_dist_from_y(left_heel_y_px, debug=True)
            right_dist = find_dist_from_y(right_heel_y_px, debug=True)
        
    # === Save to CSV (✅ append only)
    # === Save to CSV (✅ append only)
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                frame_Index,
                adjusted_time_ms,
                left_heel_y_norm,
                right_heel_y_norm,
                left_dist,
                right_dist
                ])
    else:
        #f.write(f"BACK IN MAIN BUT NOT GREAT for frame: {videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)}")
        #f.write("\n")
        min_height = 0
        max_height = height
        min_width = 0
        max_width = width
        
    resizedFrame = cv2.resize(raw_frame, displayRez) # Resize the frame for displayd
    cv2.imshow("Frame", resizedFrame) #displays frame
    key1 = cv2.waitKey(0) # Wait for a key press
    key2 = 0
    if key1 == ord('q') or key2 == ord('q') & 0xFF: exit()
