####
#   STARS Summer 2025
#   Dr J Lab
###
# Label Vibration Data with walking pace from camera
####

modelDir = r"C:\Users\smitt\STARS\\" #Kara
#modelDir = "../media-pipeModels/"   #Josh
#modelDir = r"C:\Users\notyo\Documents\STARS\mediapipe\\" #Jack

vidDir = r"E:\STARS" #Kara
#vidDir = r"C:\Users\notyo\Documents\STARS" #Jack

##
# Pause = Space, 
# Forward = g
#Backwards = d
#Backwards by 1 second = s

## Imports
#Built ins
#import time
#import datetime

#Third party
import cv2 # opencv-python
import pytesseract # pip install pytesseract
import matplotlib as plt # matplotlib
import numpy as np # numpy
import csv

# Media Pipe
import mediapipe as mp  # pip install mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys 
import os
from vibDataChunker import vibDataWindow
#import keyboard

# Our stuff
from velocity import calculate_avg_landMark_velocity 
from cv2Utils import overlay_image
from vibDataChunker import vibDataWindow

# === Fix import path to reach distance_position.py ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_position import find_dist_from_y  # ✅ Import your custom function

from OCR_Detect import timeWith_ms # Import the timeWith_ms class from OCR_Detect.py

Runthrough = False 
Playback = True 

#North_South Runs
#Kara's video file
#dir = r"StudentData\25_06_18\subject_1"
#file = r"Sub_1_Run_3__6-18-2025_11-49-29 AM.asf"
#Yoko's video file
#dir = r"StudentData\25_06_18\subject_3"
#file = r"sub_3_run_4_F_6-18-2025_11-26-08 AM.asf"
#file = r"Sub3_run6_6-18-2025_11-32-05 AM.asf"
#Jack's video file
#dir = r"StudentData\25_06_18\subject_2"
#file = r"sub_2_run_4_6-18-2025_11-41-35 AM.asf"
#Bad Run:
#dir = r"StudentData\25_06_18\subject_2"
#file = r"sub_2_run_1_6-18-2025_11-36-03 AM.asf"

#South_North Runs
#Kara's video file
#dir = r"StudentData\25_06_18\subject_1"
#file = r"Sub_1_Run_2__6-18-2025_11-47-57 AM.asf"
#Yoko's video file
#dir = r"StudentData\25_06_18\subject_3"
#file = r"Sub3_run7_6-18-2025_11-34-22 AM.asf"
#Jack's video file
#dir = r"StudentData\25_06_18\subject_2"
#file = r"sub_2_run_5_6-18-2025_11-42-48 AM.asf"
#Bad Run:
#dir = r"StudentData\25_06_18\subject_1"
#file = r"Sub_1_Run_1_6-18-2025_11-45-46 AM.asf"

#Kara's video file
#dir = r"StudentData\25_06_18\subject_1"
#file = r"Sub_1_Run_1_6-18-2025_11-45-46 AM.asf"
#file = r"Sub_1_Run_2__6-18-2025_11-47-57 AM.asf"
#file = r"Sub_1_Run_3__6-18-2025_11-49-29 AM.asf"

#Jack's video file
dir = r"StudentData\25_07_10\subject_2"
#dir = r"StudentData\25_06_18\subject_2"
#file = r"intercept_run_7-10-2025_10-45-46 AM.asf"
file = r"poll_run_7-10-2025_10-50-56 AM.asf"
#file = r"sub_2_run_3_pt_1_6-18-2025_11-40-17 AM.asf"
#file = r"sub_2_run_4_6-18-2025_11-41-35 AM.asf"
#file = r"sub_2_run_5_6-18-2025_11-42-48 AM.asf"

#Yoko's video file
#dir = r"StudentData\25_06_18\subject_3"
#file = r"sub_3_run_4_F_6-18-2025_11-26-08 AM.asf"
#file = r"sub3_run5_6-18-2025_11-28-28 AM.asf"
#file = r"Sub3_run6_6-18-2025_11-32-05 AM.asf"
#file = r"Sub3_run7_6-18-2025_11-34-22 AM.asf"

#pollvintercept Jack runs
dir = r"StudentData\25_07-10"
file = r"intercept_run_7-10-2025_10-45-46 AM.asf"
#file = r"poll_run_7-10-2025_10-50-56 AM.asf"

#dir = r"E:\STARS\07_10_2025_Vid_Data"
#file = "intercept_run_7-10-2025_10-45-46 AM.asf"
fileName = f"{vidDir}/{dir}/{file}"
print(f"Opening video: {fileName}")

# ===== Global variables

videoOpbject = cv2.VideoCapture(fileName) #open the video file and make a video object
if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()
# Video properties    
fps = 30 # Frames per second
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT) #Frame count
width = int(videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH)) # Width of the video frame
height = int(videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of the video frame
frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
dispFact = 2
displayRez = (int(width/dispFact), int(height/dispFact))
displayRezsquare = (int(height/dispFact), int(height/dispFact)) 

#vibration properties
vib = vibDataWindow(
    dir_path=r"E:\STARS\StudentData\25_07_10\subject_2",
    data_file=r"Jack_clockTest_interuptVPoll.hdf5",
    trial_to_plot=1,
    old_data=False,
    window=5
)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(r"E:\STARS\StudentData\Exported_Video\annotated_output.mp4", fourcc, fps, displayRez)
outCrop = cv2.VideoWriter(r"E:\STARS\StudentData\Exported_Video\annotated_output_crop.mp4", fourcc, fps, displayRezsquare)


# Define video writers (90-frame clip, initialized when needed)
out_full = None
out_crop = None

clip_start = 0  # Example: clip starts at frame 200
clip_length = int(fCount)  # Length of the clip in frames
clip_end = clip_start + clip_length
maintain_height_max = height
maintain_height_min = 0
maintain_width_max = width
maintain_width_min = 0


fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_dir = r"C:\Users\notyo\Documents\STARS\NSF_Floor_Vib_Camera-Labeling\NSF_Floor_Vib_Camera-Labeling\Jack\trialData"  # Set your own output path

#=== Setting up mediapipe
## Configurations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_lite.task" # 5.5 MiB
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_full.task" # 9.0 MiB
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_heavy.task" # 29.2 MiB

#model_path = r"../media-pipeModels/pose_landmarker_lite.task" # 5.5 MiB
model_path = f"{modelDir}pose_landmarker_heavy.task" # 29.2 MiB
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
                                output_segmentation_masks=True
                               )

landmarkerVideo = PoseLandmarker.create_from_options(options)


# === Functions

def findPixfromDist(distance):
    pixels = 7916.1069/(distance -1.0263) -86.1396
    return pixels

def get_key(delay=0):
    key1 = cv2.waitKey(delay)

    if key1 == 255:
        return None  # No key was pressed (in non-blocking mode)

    # Windows special key handling
    if key1 in [0, 224]:
        key2 = cv2.waitKey(0) & 0xFF
        return {
            72: 'up',
            80: 'down',
            75: 'left',
            77: 'right'
        }.get(key2, None)

    # Unix-style arrow keys and normal keys
    return {
        27: 'esc',
        ord('q'): 'q',
        81: 'left',   # ← on Unix
        82: 'up',     # ↑
        83: 'right',  # →
        84: 'down'    # ↓
    }.get(key1, chr(key1) if 32 <= key1 <= 126 else None)

# === OCR timestamp function ===
def getDateTime(frame):
    dateTime_img = frame[0:46, 0:384, :]
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)
    dateTime_img_bw = 255 - dateTime_img_bw
    data = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    try:
        #date_str = data['text'][4]
        time_str = data['text'][5]
        AM_PM = data['text'][6]
        return f"{time_str}.{AM_PM}"
    except:
        return "OCR Error"
    
def drawLandmark_circle(frame, landmark, color, radius):
    radius = radius
    thickness = -1
    frame_height, frame_width = frame.shape[:2]
    #color = [255, 0, 0] #Circle will be red
    #center = int(center_width), int(center_height)
    center = int(landmark.x*frame_width), int(landmark.y*frame_height) #place center of the circle at the landmark position
    #print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")q
    cv2.circle(frame, center, radius, color, thickness) # Draw a circle at the landmark position

def addLandmark_text(frame, landmark, text, color):
    frame_height, frame_width = frame.shape[:2]
    #color = [255, 0, 0] #Circle will be red
    center = int(landmark.x*frame_width), int(landmark.y*frame_height) #place center of the circle at the landmark position
    #print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")
    cv2.putText(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def drawLandmark_line(frame, feet, hips, color):
    #color = [255, 0, 0] # Line will be red
    pt1_ft = (int(feet.x*width),int(feet.y*height)) #First point is on the feet
    pt2_hips = (int(hips.x*width), int(hips.y*height)) #second point is on the hips
    thickness = 5
    cv2.line(frame,pt1_ft,pt2_hips, color, thickness) # Draw a line from the feet to the hips
   
def drawLandmark_square(frame, landmark, color, thickness = -1):
    x_min = int(landmark.x*width -10)
    y_min = int(landmark.y*height -10)
    x_max = int(landmark.x*width +10)
    y_max = int(landmark.y*height +10)
    pt1 = x_min,y_max
    pt2 = x_max,y_min
    cv2.rectangle(frame,pt1,pt2,color,thickness)

def isPersonInFrame(frame, frameIndex): #(frame, frameIndex)
    """
    Checks for pose landmarks in the given frame.

    Returns:
        pose_results: A list of objects containing pose landmarks
        frame_timestamp_ms: The timestamp of the frame in milliseconds
        bool: True if pose landmarks are detected, False otherwise
    """
    frame_timestamp_ms = int(frameIndex * frameTime_ms) 
    if frame_timestamp_ms < 0 or frame_timestamp_ms > 1e10: # Check if the timestamp is valid
        return None #Exit function if the timestamp is invalid
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarkerVideo.detect_for_video(mp_image, frame_timestamp_ms) #Detect the pose landmarks in the frame
    #If there are pose landmarkers return them and the frame timestamp
    if len(pose_landmarker_result.pose_landmarks) > 0: 
        return True, pose_landmarker_result, frame_timestamp_ms
    else:
        return False, None, frame_timestamp_ms

def crop_to_Southhall():

    """
    Crops the frame to the South Hallway dimensions.

    Returns:
        dimensions: min_width, max_width, min_height, max_height --> encapsulates the South Hallway
    """
    min_height = 0
    max_height = 254
    adjust_width = (max_height - min_height) //2
    center_width = width//2 + 60
    min_width = center_width - adjust_width
    max_width = center_width + adjust_width
    direction  = "South"
    return round(min_width), round(max_width), round(min_height), round(max_height), direction 

def crop_to_Northhall():
    """
    Crops the frame to the North Hallway dimensions.

    Returns:
        dimensions: min_width, max_width, min_height, max_height --> encapsulates the North Hallway
    """
    min_height = 0
    max_height = height
    adjust_width = (max_height - min_height) //2
    center_width = width//2
    min_width = center_width - adjust_width
    max_width = center_width + adjust_width
    return round(min_width), round(max_width), round(min_height), round(max_height), direction

def blur_person_fullFrame(raw_frame, newDim_Frame, landmark, min_height, max_height, min_width, max_width):
    """
    Returns a full-size frame with the person blurred and the background untouched.

    Parameters:
        raw_frame: Full-size original image
        newDim_Frame: Cropped frame (where person is detected)
        landmark: pose landmarks
        dimensions: min_height, max_height, min_width, max_width --> Coordinates of the crop in full frame

    Returns:
        raw_frame: raw_frame-sized image with the person blurred
    """
    if result.segmentation_masks is None:
        return raw_frame
    # Resize the segmentation mask to match the cropped region
    #Saves segmentation mask as numpy array
    crop_mask = landmark.segmentation_masks[0].numpy_view() 
    #Resize the numpy array to match the cropped frame size.
    crop_mask = cv2.resize(crop_mask, (newDim_Frame.shape[1], newDim_Frame.shape[0])) 
    #Creates a boolean array that tracks where a person is likely located (> .5)
    #Converts the boolean values to unsigned integers and scales values to binary image format (white = 255 = person = true)
    binary_crop_mask = (crop_mask > 0.5).astype(np.uint8) * 255  
    #creates a NumPy array of zeros that matches the height and width of the raw_frame
    full_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
    #pastes the binary_crop_mask into the area in the full frame where the cropped frame originally came from. 
    full_mask[min_height:max_height, min_width:max_width] = binary_crop_mask
    #Inverts mask (person becomes black and background is white)
    inverse_mask = cv2.bitwise_not(full_mask)
    #Blurs full frame, The values 61, 61 can be adjusted to increase/decrease blur
    blurred_frame = cv2.GaussianBlur(raw_frame, (61, 61), 0)
    #Saves blurred pixels in locations where full_mask is non-zero (The location of person)
    person_blurred = cv2.bitwise_and(blurred_frame, blurred_frame, mask=full_mask)
    #Saves unedited pixels in locations where inverse_mask is non-zero (The location of background)
    background_clear = cv2.bitwise_and(raw_frame, raw_frame, mask=inverse_mask)
    #Adds pixels to create a frame with a blurred person and unedited background
    raw_frame = cv2.add(person_blurred, background_clear)
    return raw_frame
    
def blur_person_cropFrame(cropped_frame, result):
    """
    Blurs only the person within a cropped frame,
    keeping the background untouched.

    Parameters:
        cropped_frame: The cropped portion of the original frame
        result: The MediaPipe result 

    Returns:
        final_cropped_frame: The cropped frame with the person blurred
    """
    if result.segmentation_masks is None:
        return cropped_frame
    #Saves segmentation mask as numpy array
    crop_mask = result.segmentation_masks[0].numpy_view() 
    #Resize the numpy array to match the cropped frame size.
    crop_mask = cv2.resize(crop_mask, (cropped_frame.shape[1], cropped_frame.shape[0])) 
    #Creates a boolean array that tracks where a person is likely located (> .5)
    #Converts the boolean values to unsigned integers and scales values to binary image format (white = 255 = person = true)
    binary_crop_mask = (crop_mask > 0.5).astype(np.uint8) * 255  
    #Inverts mask (person becomes black and background is white)
    inverse_mask = cv2.bitwise_not(binary_crop_mask)
    #Blurs frame, The values 61, 61 can be adjusted to increase/decrease blur
    blurred_crop = cv2.GaussianBlur(cropped_frame, (61, 61), 0)
    #Saves blurred pixels in locations where blurred_crop is non-zero (The location of person)
    person_blurred = cv2.bitwise_and(blurred_crop, blurred_crop, mask=binary_crop_mask)
    #Saves unedited pixels in locations where cropped_frame is non-zero (The location of background)
    background_clear = cv2.bitwise_and(cropped_frame, cropped_frame, mask=inverse_mask)
    #Adds pixels to create a frame with a blurred person and unedited background
    final_cropped_frame = cv2.add(person_blurred, background_clear)
    return final_cropped_frame

def crop_to_square(frame, landmarks, direction, maintain_dim):
    """
    Crops the frame based on the pose landmarks in a scaled square manner 

    Parameters:
        frame: The original, uncropped frame
        landmark: The MediaPipe result containing the pose landmarks
        direction: The direction  person is starting in the hallway (North or South), initially expects North
        maintain_dim: Originally set to full frame dimensions

    Returns:
        dimension: min_width, max_width, min_height, max_height  --> Dimensions to crop the frame around person w/square   
        maintain_dim: Updated dimensions to maintain square proportions when the person is nearing the edge of the frame
    """
    #Checks if there are landmarkers (if none, return full frame)
    if landmarks is None:
        return 0, frame.shape[1], 0, frame.shape[0], maintain_dim #frame.shape[1] = width, frame.shape[0] = height
    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2] 
    # Use only major body parts that are symmetrical and close to the torso
    core_landmarks = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]  # shoulders, hips, knees, etc.
    #Initiates width/height variables to landmark 0
    min_width = max_width = landmarks[0].x 
    min_height = max_height = landmarks[0].y
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
    #Normalize landmark values (0-1) to frame dimensions
    min_width = min_width*frame_width
    max_width = max_width*frame_width
    min_height=min_height*frame_height
    max_height=max_height*frame_height
    #total width/height of cropped frame
    tot_width = max_width - min_width  
    tot_height = max_height - min_height 
    #Finds the center WRT full frame by locating the center in the full frame first (W or H/ 2)
    #then expanding it by minimum width/height of cropped frame
    center_width = min_width + tot_width / 2 
    center_height = min_height + tot_height / 2
    #Adjusts width to be square
    adjust_width = tot_height//2
    min_width = center_width - adjust_width
    max_width = center_width + adjust_width
    tot_width = max_width - min_width   #total width of square frame
    #Expand the crop to include more of the frame
    scale_factor = 1.6
    #scales total width/height
    new_width = tot_width * scale_factor
    new_height = tot_height * scale_factor
    #Calculates new min/max height/width WRT to full frame by starting from the center of the full frame 
    #expanding by the width/height of the scaled square frame
    min_width = center_width - new_width / 2
    max_width = center_width + new_width / 2
    min_height = center_height - new_height / 2
    max_height = center_height + new_height / 2
    #Maintains square dimensions when the person is nearing the edge of the frame
    if direction  == "South": 
        #If the maximum height is within 1% of the original height, maintain the dimensions
        #Since the height is rarely ever equal to the original height, this is a failsafe
        #if statement should be true when the person is nearing the northmost part of the hallway
        if abs(max_height - height) / height <= .01:
            #print("Maintaining South Hallway")
            maintain_dim[1] = height
            maintain_dim[0] = min_height
            maintain_dim[3] = max_width
            maintain_dim[2] = min_width
        #Ensures the crop of the person is within bounds at the southmost part of the hallway
        elif min_height < 0:
            min_height = 0
        #Ensures the crop of the person is within bounds and square at the northmost part of the hallway
        elif max_height > height:
            #print("Resetting South Hallway")
            max_height = maintain_dim[1]
            min_height = maintain_dim[0]
            max_width = maintain_dim[3]
            min_width = maintain_dim[2]
            
    #Maintains square dimensions when the person is nearing the edge of the frame
    if direction  == "North":
        #If the minimum height is within 5 pixels of the original height, maintain the dimensions
        if abs(min_height) <= 5:
            maintain_dim[1] = max_height
            maintain_dim[0] = 0 
            maintain_dim[3] = max_width
            maintain_dim[2] = min_width
        #Ensures the crop of the person is within bounds at the northmost part of the hallway (repeat of crop_to_Northhall)
        elif max_height > height:
            max_height = height
            min_height = 0
            adjust_width = (max_height - min_height) //2
            center_width = width//2
            min_width = center_width - adjust_width
            max_width = center_width + adjust_width
        #Ensures the crop of the person is within bounds and square at the southmost part of the hallway
        elif min_height < 0:
            max_height = maintain_dim[1]
            min_height = maintain_dim[0]
            max_width = maintain_dim[3]
            min_width = maintain_dim[2]
    # Make sure the result isn't an empty crop
    if max_width <= min_width or max_height <= min_height:
    # Return the full frame as fallback
        return 0, frame_width, 0, frame_height, maintain_dim
    return round(min_width), round(max_width), round(min_height), round(max_height), maintain_dim

def smooth_crop_dim(smoothed_dim, min_width, max_width, min_height, max_height, alpha=0.1):
    """
    Smooths the crop dimensions using exponential smoothing.

    Parameters:
        Dimensions: min_width, max_width, min_height, max_height --> cropped frame
        alpha: Smoothing factor (between 0 and 1)  --> Higher values mean less smoothing
        smoothed_dim: List to store the smoothed dimensions  --> (Expects [None, None, None, None] initially)
    Returns:
        smoothed_dim: [smoothed_min_h, smoothed_max_h, smoothed_min_w, smoothed_max_w] --> Smoothed crop dimensions
        smoothCrop_dim: min_width, max_width, min_height, max_height --> crop dimensions based on smoothing
    """
    #Initializes smoothed_dim to original dimensions if None
    if smoothed_dim[0] is None:
        smoothed_dim[0] = min_height
        smoothed_dim[1] = max_height
        smoothed_dim[2] = min_width
        smoothed_dim[3] = max_width
    # Apply exponential smoothing to the dimensions
    else:
        smoothed_dim[0]  = int(alpha * min_height + (1 - alpha) * smoothed_dim[0])
        smoothed_dim[1] = int(alpha * max_height + (1 - alpha) * smoothed_dim[1])
        smoothed_dim[2] = int(alpha * min_width + (1 - alpha) * smoothed_dim[2])
        smoothed_dim[3] = int(alpha * max_width + (1 - alpha) * smoothed_dim[3])
    
    min_height, max_height = smoothed_dim[0], smoothed_dim[1] 
    min_width, max_width = smoothed_dim[2], smoothed_dim[3]   
    return (smoothed_dim, min_width, max_width, min_height, max_height)
    
def landmarks_of_fullscreen(landmarks, min_width, max_width, min_height, max_height, width = width, height = height):
    """
    Takes landmarks in cropped frame dimensions and expands them to full frame dimensions.

    Parameters:
        Dimensions: min_width, max_width, min_height, max_height --> cropped frame
        Fullframe_Dim :width, height: width and height of the full frame --> defaults to global width and height
        Landmark: The MediaPipe result containing the pose landmarks 

    Returns:
        Nothing. The landmarks are mutable and are modified in place.
    """
    full_width = width
    full_height = height
    for i in range(len(landmarks)):   
        landmarks[i].x = (landmarks[i].x * (max_width - min_width) + min_width) / full_width
        landmarks[i].y = (landmarks[i].y * (max_height - min_height) + min_height) / full_height

def seconds_sinceMidnight(timeWith_ms_class,raw_frame):
    #Get seconds from midnight from the frame timestamp
    timestamp = getDateTime(raw_frame) # Get the timestamp from the frame
    HHMMSS, AM_PM = timestamp.split('.') # Split the timestamp into time and AM/PM
    timestamp_withms = time_tracker.calc_ms(HHMMSS, frame_Index) # Get the timestamp with milliseconds
    hours, minutes, seconds = timestamp_withms.split(':') # Split the timestamp into hours, minutes, seconds
    seconds, milliseconds = seconds.split('.') # Split seconds into seconds and milliseconds
    if AM_PM == "PM" and hours != "12": # Convert PM to 24-hour format
        hours = str(int(hours) + 12)
    elif AM_PM == "AM" and hours == "12": # Convert 12 AM to 00 hours
        hours = "00"
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000 # Convert to total seconds
    return total_seconds # Return the total seconds since midnight

def playback(frame_array, frameIndex):
    pass

def create_Trackframes(firstframe, lastframe, *definitions):
    """
    Initialises a list with dictionary definitions 
    
    Parameters:
    firstframe: frame to start
    lastframe: frame to end 
    definitions: any dictionary items the user want to input (Will be initialised with None)

    Returns:
    track_frames: A list length of the total frames to be played with dictionary definitions
    """
    default_dict = {key: None for key in definitions} # List to store all cropped frames
    track_frames = [default_dict.copy() for _ in range(lastframe - firstframe)] # Initialize the list with the number of frames
    return track_frames

def put_text(text_array, frame):
    """
    Adds text from a dictionary to a frame with the same index 
    
    Parameters:
    text_array: An array of strings
    frame_array: An array of frames and information about walking pace associated
    with each frame
    frame_arrayIndex: The index of the frame to put text

    Returns:
    Nothing. The frames are mutable
    """
    text_arrayIndex = 0
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.7
    thickness = 1
    x, y = 10, 100  #Coordinates initialised to top-left corner of frame
    while text_arrayIndex < len(text_array): #Iterate through text_aray
        words = text_array[text_arrayIndex] #Save string at index to words
        (text_width, text_height), baseline = cv2.getTextSize(words, font, scale, thickness) #
        cv2.rectangle(
        frame,
        (x - 2, y - text_height - 2),                # Top-left corner
        (x + text_width + 2, y + baseline + 2),      # Bottom-right corner
        (255, 255, 255),                             # White background
        thickness=cv2.FILLED                         # Filled rectangle
    )
        cv2.putText(frame, text_array[text_arrayIndex], (x, y), font, scale, (0,0,0), thickness)
        text_arrayIndex = text_arrayIndex+1
        y = y + (text_height +20)


def grow_constantSize(landmarks, size_cm, frame_I, start_F, end_F, prev_px=None, alpha=0.1):
    y_pix_height = landmarks.y* height
    #distance_from_cam = 7916.1069 / (y_pix_height + 86.1396) - 1.0263
    total_frames = end_F - start_F 
    cm_per_px = (7916.1069 / (y_pix_height + 86.1396)**2) * 100
    raw_px = size_cm / cm_per_px
    # Normalize frame progress between 0 and 1
    progress = (frame_I - start_F) / total_frames
    progress = max(0, min(progress, 1))  # Ensures frame_index is within range
    # Define a target multiplier as a function of frame progress
    # For example: linearly increases from 1 to 4 as progress goes from 0 to 1
    target_multiplier = 1 + 16 * progress
    target_px = raw_px * target_multiplier

    # If this is the first frame, no previous px to smooth from
    if prev_px is None:
        smoothed_px = target_px
    else:
        smoothed_px = alpha * target_px + (1 - alpha) * prev_px

    return int(smoothed_px), smoothed_px

def constantSize(landmarks, size_cm, frame_I, start_F, end_F, prev_px=None, alpha=0.1):
    y_pix_height = landmarks.y* height
    #distance_from_cam = 7916.1069 / (y_pix_height + 86.1396) - 1.0263
    total_frames = end_F - start_F 
    cm_per_px = (7916.1069 / (y_pix_height + 86.1396)**2) * 100
    raw_px = size_cm / cm_per_px
    # Normalize frame progress between 0 and 1
    progress = (frame_I - start_F) / total_frames
    progress = max(0, min(progress, 1))  # Ensures frame_index is within range

    # If this is the first frame, no previous px to smooth from
    if prev_px is None:
        smoothed_px = raw_px
    else:
        smoothed_px = alpha * raw_px + (1 - alpha) * prev_px

    return int(smoothed_px), smoothed_px

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

# === Main code === #

# === Set time to start/end
start_time = 0

start_frame = int(start_time * fps) # Start frame for the clip
end_time = 30 # End time for the clip in seconds
end_frame = fps*end_time #int(fCount)
# === saves dimensions for first crop
max_height = height
min_height = 0
max_width = width
min_width = 0
# === Initialise arrays and variables to send to functions
maintain_dim = [0,height,0,width] # Initializes maintain dimensions
smoothed_dim = [None, None, None, None]  # Initialize smoothed dimensions -> [smoothed_min_h, smoothed_max_h, smoothed_min_w, smoothed_max_w]
time_tracker = timeWith_ms(frameTime_ms) #Creates object
alpha = .1  # smoothing factor between 0 (slow) and 1 (no smoothing)
direction  = "North" #Default direction 
track_frames = create_Trackframes(start_frame, end_frame, "frame", "cropped_frame", "landmarks",
                                  "LeftToe_Dist","RightToe_Dist", "RightHeel_Dist", "LeftHeel_Dist", 
                                 "seconds_sinceMid", "toeVel", "heelVel") #array to track information about each frame
crop_prevPixR_Toe = None
crop_prevPixL_Toe = None
crop_prevPixR_Heel = None
crop_prevPixL_Heel = None                                 
prevPixR_Toe = None
prevPixL_Toe = None
prevPixR_Heel = None
prevPixL_Heel = None
windowLen_s = 1 #5
windowInc_s = 0.5 #1
pixel_incm = 6
cropped_pixel_incm = 3


# === Write to file
csv_path = r"C:\Users\notyo\Documents\STARS\NSF_Floor_Vib_Camera-Labeling\NSF_Floor_Vib_Camera-Labeling\Jack\trialData\footstep_data_poll.csv"
with open(csv_path, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([
                "Time", "Seconds_Mid",
        "LeftHeel_Dist", "RightHeel_Dist" , 
        "LeftToe_Dist" , "RightToe_Dist",
    ]) 


# === Prompt for user
print(f"Press f to pause the video then you will be able to use other keys to navigate through the video frames. Press q to quit.")

# === Sets the video to specified index
frame_Index = start_frame
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, frame_Index)

# === Begin process of cropping, saving, and playback
waitKeyP = 1
toeVel_mps = 0
framewith_data = 0

vibImage_rgba = None
windowName = "Main Frame:"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

success, raw_frame = videoOpbject.read() # Returns a boolean and the next frame
initial_seconds = seconds_sinceMidnight(time_tracker, raw_frame)

print(f"Initial seconds: {initial_seconds}")

videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, frame_Index)

while frame_Index < end_frame:
    i = frame_Index - start_frame #index for track_frames array
    # === Reads and loads new frames in array
    if track_frames[i]['frame'] is None: 
        #print(f"frame_Index: {frame_Index}, i: {i}")
        success, raw_frame = videoOpbject.read() # Returns a boolean and the next frame
        if not success: # If the frame was not read successfully, break the loop
            print("Failed to read frame")
            exit()
        # === Saves seconds since midnight
        total_seconds = seconds_sinceMidnight(time_tracker, raw_frame) 
        print(f"Total seconds: {total_seconds}")
        # === Crops full frame. Draws the cropped area on full frame
        newDim_Frame = raw_frame[min_height:max_height,min_width:max_width,:].copy() #crops frame
        cv2.rectangle(raw_frame, (min_width,max_height), (max_width, min_height), [255,0,0], 5)
        # ===Is there a cropped frame to send to model?
        if newDim_Frame is not None: 
            good = False
        # === Returns landmarks based on person
            good, result, adjusted_time_ms = isPersonInFrame(newDim_Frame, frame_Index)
        # === 
            if good and result is not None:
                landmarks = result.pose_landmarks[0]
                constPixL_Toe, crop_prevPixL_Toe = constantSize(landmarks[31],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixL_Toe)
                drawLandmark_circle(newDim_Frame, landmarks[31], [230, 216, 173], constPixL_Toe) #left toe is light blue
                constPixL_Heel, crop_prevPixL_Heel = constantSize(landmarks[29],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixL_Heel)
                drawLandmark_circle(newDim_Frame, landmarks[29], [139, 0, 0],constPixL_Heel) # left heel is dark blue
                constPixR_Heel, crop_prevPixR_Heel = constantSize(landmarks[32],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixR_Heel)
                drawLandmark_circle(newDim_Frame, landmarks[32], [102, 102, 255],constPixR_Heel) # right toe is light red 
                constPixR_Toe, crop_prevPixR_Toe = constantSize(landmarks[30],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixR_Heel)
                drawLandmark_circle(newDim_Frame, landmarks[30], [0, 0, 139],constPixR_Toe) #right heel is dark red

                landmarks_of_fullscreen(landmarks, min_width, max_width, min_height, max_height) 
                #=== Draws landmarks and expands them according to pixel size
                
                constPixL_Toe, prevPixL_Toe = constantSize(landmarks[31],pixel_incm, frame_Index, start_frame, end_frame, prevPixL_Toe)
                drawLandmark_circle(raw_frame, landmarks[31], [230, 216, 173],constPixL_Toe) #left toe is light blue
                constPixL_Heel, prevPixL_Heel = constantSize(landmarks[29],pixel_incm, frame_Index, start_frame, end_frame, prevPixL_Heel)
                drawLandmark_circle(raw_frame, landmarks[29], [139, 0, 0],constPixL_Heel) # left heel is dark blue
                constPixR_Heel, prevPixR_Heel = constantSize(landmarks[32],pixel_incm, frame_Index, start_frame, end_frame, prevPixR_Heel)
                drawLandmark_circle(raw_frame, landmarks[32], [102, 102, 255],constPixR_Heel) # right toe is light red 
                constPixR_Toe, prevPixR_Toe = constantSize(landmarks[30],pixel_incm, frame_Index, start_frame, end_frame, prevPixR_Heel)
                drawLandmark_circle(raw_frame, landmarks[30], [0, 0, 139],constPixR_Heel) #right heel is dark red 
                # === Get new frame dimensions           
                min_width, max_width, min_height, max_height, maintain_dim  = crop_to_square(raw_frame, landmarks, direction ,maintain_dim) 
                smoothed_dim, min_width, max_width, min_height, max_height  = smooth_crop_dim(smoothed_dim, min_width, max_width, min_height, max_height) 
                # === Saves data to array
                track_frames[i]["landmarks"] = landmarks # Store the landmarks in the track_frames list
                    # === Calculates distance
                left_distHeel = find_dist_from_y(track_frames[i]["landmarks"][29].y*height)
                right_distHeel = find_dist_from_y(track_frames[i]["landmarks"][30].y*height)
                left_distToe = find_dist_from_y(track_frames[i]["landmarks"][31].y*height)
                right_distToe = find_dist_from_y(track_frames[i]["landmarks"][32].y*height)

                track_frames[i]["LeftToe_Dist"] = left_distToe # Store the left toe distance in the track_frames list
                track_frames[i]["RightToe_Dist"] = right_distToe # Store the left toe distance in the track_frames list
                track_frames[i]["RightHeel_Dist"] = right_distHeel
                track_frames[i]["LeftHeel_Dist"] = left_distHeel 
                
                track_frames[i]["seconds_sinceMid"] = safe_divide(i, fps)
                # Calculate the walking speed 
                # Every n seconds (how many frames is that)
                if framewith_data >= (windowLen_s+1)*fps:    # don't run if we don't have a windows worth of data
                                                # Also, skip the times that don't have rollovers
                    if framewith_data % (windowInc_s*fps) == 0: # run every overlap
                        #print(f"Calculate ms at frame: {i}, fps:{fps}, inc: {windowInc_s} sec")
                        #print(f"distance: {track_frames[i]["LeftToe_Dist"]}, landmark: {track_frames[i]["landmarks"][29].y}")
                        heelVel_mps = calculate_avg_landMark_velocity(track_frames, left="LeftHeel_Dist", right="RightHeel_Dist", curentFrame=i, nPoints= windowLen_s*fps, verbose=False)
                        toeVel_mps = calculate_avg_landMark_velocity(track_frames, left="LeftToe_Dist", right="RightToe_Dist", curentFrame=i, nPoints= windowLen_s*fps, verbose=False)

                        # TODO:Jack Get vibration data

                        # send time  seconds since midnight and location of walker
                        # returns:  img_rgba = np.asarray(canvas.buffer_rgba())
                        vibImage_rgba = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[1-1])
                        vibImage_rgba2 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[2-1], colVar ='blue')
                        vibImage_rgba3 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[3-1], colVar ='orange')
                        vibImage_rgba4 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[4-1], colVar ='darkgoldenrod')
                        vibImage_rgba5 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[5-1], colVar ='green')
                        vibImage_rgba6 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[6-1], colVar ='purple')
                        vibImage_rgba7 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[7-1], colVar ='olive')

                        


                if vibImage_rgba is not None:
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba, loc_x=950, loc_y=int(findPixfromDist(7.59)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba2, loc_x=1900, loc_y=int(findPixfromDist(10.26)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba3, loc_x=1000, loc_y=int(findPixfromDist(12.32)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba4, loc_x=1750, loc_y=int(findPixfromDist(13.99)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba5, loc_x=1050, loc_y=int(findPixfromDist(17)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba6, loc_x=1600, loc_y=int(findPixfromDist(23.32)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba7, loc_x=1100, loc_y=int(findPixfromDist(26.82)), dim_x=200, dim_y=200) # overlay at this position
                    

                track_frames[i]["toeVel"] = toeVel_mps
                track_frames[i]["heelVel"] = toeVel_mps

                text = [
                    f"Seconds: {track_frames[i]["seconds_sinceMid"]:.3f} s",
                    f"Left Heel: {track_frames[i]["LeftHeel_Dist"]:.2f} m", 
                    f"Left Toe: {track_frames[i]["LeftToe_Dist"]:.2f} m", 
                    f"Right Heel: {track_frames[i]["RightHeel_Dist"]:.2f} m",
                    f"Right Toe: {track_frames[i]["RightToe_Dist"]:.2f} m",
                    "Previous Window: ",
                    f"Toe Vel: {track_frames[i]["toeVel"]:.2f} m/s",
                    f"Heel Vel: {track_frames[i]["heelVel"]:.2f} m/s",
                    ]
                framewith_data +=1

                # TODO: Add vibration data to frame
            else: # not good or no result
                text = [
                    ]
                if frame_Index % 2 ==0:
                    min_width, max_width, min_height, max_height, direction = crop_to_Southhall() #, landmarks
                else:
                    min_width, max_width, min_height, max_height, direction = crop_to_Northhall() #, landmarks
            resized_rawframe = cv2.resize(raw_frame, displayRez)
            resizedframe = cv2.resize(newDim_Frame, displayRezsquare)
            # ===resize for viewing and save in array
            put_text(text,  resizedframe)
            put_text(text, resized_rawframe)
            #print(f"shape | raw_frame {raw_frame.shape}, resized_rawframe {resized_rawframe.shape}")
            track_frames[i]["frame"] = resized_rawframe
            track_frames[i]["cropped_frame"] = resizedframe
            out.write(resized_rawframe)  # Save the frame to video
            outCrop.write(resizedframe)  # Save the frame to video

    else:
        resized_rawframe = track_frames[i]["frame"]
        resizedframe = track_frames[i]["cropped_frame"]

    cv2.imshow("Zoomed Frame: ", resizedframe)
    cv2.imshow("Frame", resized_rawframe)
    #cv2.resizeWindow(windowName, 1433, 756) #TODO: use from vars

    # Navigation
    key1 = cv2.waitKey(waitKeyP) #& 0xFF  
    #key1 = get_key(waitKeyP)
    #print(f"key: {key1}")

    if key1 == 32: #Space to pause
        if waitKeyP == 1:
            waitKeyP = 0
            print("Pausing") 
        else:
            frame_Index -= 1 # when we unpause we will increment, but that will skip on
            waitKeyP = 1
            print("Resuming") 
            frame_Index = frame_Index + 1
    elif key1 == 81 or key1 ==2 or key1 == ord('d'): #Left Arrow:  # Back one Frame
        waitKeyP = 0 # If we key we want to pause
        frame_Index -= 1
        if frame_Index < start_frame:
            print("Cannot go further back, press space to continue")
            frame_Index = start_frame
    elif key1 == 84 or key1 == 1 or key1 == ord('s'):  # Down Arrow Back one Second
        #print(f"back one second: {fps} frames")
        waitKeyP = 0
        frame_Index -= fps
        if frame_Index < start_frame:
            print("Cannot go further back, press space to continue")
            frame_Index = start_frame
    elif key1 == 83 or key1 == 3 or key1 == ord('g'):  #Right Arrrow Step forwared One Frame
        #print(f"Forward one frame")
        waitKeyP = 0 # If we key we want to pause
        frame_Index += 1 
        if (frame_Index - start_frame) >= len(track_frames):
            #print("Reached the end of video")
            frame_Index -= 1 
            #continue             
    elif key1 == 82 or key1 == 0 or key1 == ord('h'):  #Up Arrow Forward one second
        #print(f"forward one second: {fps} frames")
        waitKeyP = 0 # If we key we want to pause
        frame_Index += fps
        #if i >= len(track_frames):
        if track_frames[frame_Index - start_frame]['frame'] is None:
            frame_Index -= fps
            print("Reached the end of buffered video")
            #continue                   
    elif key1 == ord('q'):
        print("Quitting.")
        exit()


    # If we are not paulsed go to the next frame
    if waitKeyP != 0: frame_Index = frame_Index + 1 
out.release()
outCrop.release()
cv2.destroyAllWindows()
        
                        
    """with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
        #frame_Index,
        total_seconds,  # Convert to seconds
        left_distHeel,
        right_distHeel,
        left_distToe,
        right_distToe
    ])
            """

    if clip_start <= frame_Index < clip_end:
        if out_full is None:
            out_full = cv2.VideoWriter(os.path.join(output_dir, "full_frame_clip.avi"), fourcc, fps, (width, height))

        out_full.write(raw_frame)

