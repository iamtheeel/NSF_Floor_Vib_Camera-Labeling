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

import datetime
# === Fix import path to reach distance_position.py ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_position import find_dist_from_y  # ✅ Import your custom function

from OCR_Detect import timeWith_ms # Import the timeWith_ms class from OCR_Detect.py

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_lite.task" #5.5 MB
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_full.task" #9.0 MB
model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_heavy.task" #29.2 MB

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_07-10'
file = 'intercept_run_7-10-2025_10-45-46 AM.asf'
fileName = f"{dir}/{file}"  # Path to the video file

#Global variables
videoOpbject = cv2.VideoCapture(fileName) #open the video file and make a video object
if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties    
fps = 30 # Frames per second
#print(f"FPS: {fps}")
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT) #Frame count
#height, width, _ = videoOpbject.read()[1].shape doing this reads the first frame, which we don't want to do yet
width = int(videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH)) # Width of the video frame
height = int(videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of the video frame
#width = 256
#height = 256 

"""
# Define video writers (90-frame clip, initialized when needed)
out_full = None
out_crop = None

clip_start = 0  # Example: clip starts at frame 300
clip_length = int(fCount)  # Length of the clip in frames
clip_end = clip_start + clip_length
maintain_height_max = height
maintain_height_min = 0
maintain_width_max = width
maintain_width_min = 0
"""

maintain_dim = [height, 0, width, 0] 

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#output_dir = r"E:\STARS\Clips"  # Set your own output path


frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
dispFact = 2
displayRez = (int(width/dispFact), int(height/dispFact))
squareDisplayRez = (int(500),int(500))

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
                                output_segmentation_masks=True
                               )

landmarkerVideo = PoseLandmarker.create_from_options(options)
#exit()

#functions


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
    direction  = "North"
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
        final_frame: raw_frame-sized image with the person blurred
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
    final_frame = cv2.add(person_blurred, background_clear)
    return final_frame
    
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

def crop_to_ratio(frame, landmarks):
    """
    Crops the frame based on the pose landmarks while maintaining the original aspect ratio of the original frame 

    Parameters:
        frame: The original, uncropped frame
        landmark: The MediaPipe result containing the pose landmarks

    Returns:
        dimension: min_width, max_width, min_height, max_height  --> Dimensions to crop the frame around person w/ aspect ratio  
    """
    #Checks if there are landmarkers (if none, return full frame)
    if landmarks is None: 
        return 0, frame.shape[1], 0, frame.shape[0] #frame.shape[1] = width, frame.shape[0] = height
    frame_height, frame_width = frame.shape[:2] 
    # Use only major body parts that are symmetrical and close to the torso
    core_landmarks = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]  # shoulders, hips, knees, etc.
    #Initiates width variables to landmark 0
    min_width = max_width = landmarks[0].x 
    #Initiates height variables to landmark 
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
    #Height/width ratio of full sized frame
    Ratio = width/height 
    #Height/width ratio of cropped frame
    current_ratio = tot_width / tot_height 
    #Finds the center WRT full frame by locating the center in the full frame first (W or H/ 2)
    #then expanding it by minimum width/height of cropped frame
    center_width = min_width + tot_width / 2 
    center_height = min_height + tot_height / 2
    #Change height/width ratio of cropped frame to match that of full frame
    if current_ratio < Ratio:
    # Too narrow: increase width (or crop height)
        adjust_width = (tot_height * Ratio) / 2
        min_width = center_width - adjust_width
        max_width = center_width + adjust_width
    else:
    # Too wide: crop width (or increase height)
        adjust_height = (tot_width / Ratio) / 2
        min_height = center_height - adjust_height
        max_height = center_height + adjust_height
    #adjusts total width/height according to new dimensions
    tot_width = max_width - min_width
    tot_height = max_height - min_height
    #adjusts center according to new dimensions
    center_width = min_width + tot_width / 2
    center_height = min_height + tot_height / 2
    #Expand the crop to include more of the frame
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

def crop_to_square(frame, landmarks, direction, maintain_dim):
    """
    Crops the frame based on the pose landmarks in a scaled square manner 

    Parameters:
        frame: The original, uncropped frame
        landmark: The MediaPipe result containing the pose landmarks
        direction: The direction  person is starting in the hallway (North or South)
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
            print("Maintaining South Hallway")
            maintain_dim[1] = height
            maintain_dim[0] = min_height
            maintain_dim[3] = max_width
            maintain_dim[2] = min_width
        #Ensures the crop of the person is within bounds at the southmost part of the hallway
        elif min_height < 0:
            min_height = 0
        #Ensures the crop of the person is within bounds and square at the northmost part of the hallway
        elif max_height > height:
            print("Resetting South Hallway")
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

def seconds_sinceMidnight(timeWith_ms_class,raw_frame, frame_Index):
    #Get seconds from midnight from the frame timestamp
    timestamp = getDateTime(raw_frame) # Get the timestamp from the frame
    HHMMSS, AM_PM = timestamp.split('.') # Split the timestamp into time and AM/PM
    timestamp_withms = time_tracker.calc_ms(HHMMSS, True) # Get the timestamp with milliseconds
    hours, minutes, seconds = timestamp_withms.split(':') # Split the timestamp into hours, minutes, seconds
    seconds, milliseconds = seconds.split('.') # Split seconds into seconds and milliseconds
    if AM_PM == "PM" and hours != "12": # Convert PM to 24-hour format
        hours = str(int(hours) + 12)
    elif AM_PM == "AM" and hours == "12": # Convert 12 AM to 00 hours
        hours = "00"
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000 # Convert to total seconds
    return total_seconds, timestamp # Return the total seconds since midnight

# Main code
start_time = 0
start_frame = int(start_time * fps) # Start frame for the clip
end_time = 30 # End time for the clip in seconds
end_frame = int(end_time * fps) # End frame for the clip
print("Initial frame position:", videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)) #Ensures initial frame is 0dd

# Read frames until we reach the frame prior to start frame
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

#initializes height/width to full size
max_height = height
min_height = 0
max_width = width
min_width = 0

maintain_dim = [0,height,0,width] # Initializes maintain dimensions
smoothed_dim = [None, None, None, None]  # Initialize smoothed dimensions -> [smoothed_min_h, smoothed_max_h, smoothed_min_w, smoothed_max_w]


csv_path = r"C:\Users\notyo\Documents\STARS\NSF_Floor_Vib_Camera-Labeling\NSF_Floor_Vib_Camera-Labeling\Jack\trialData\sub2_071025_intercept.csv"
with open(csv_path, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([
        "Time", "Seconds_Mid",
        "LeftHeel_Dist", "RightHeel_Dist" , 
        "LeftToe_Dist" , "RightToe_Dist",
    ]) 


time_tracker = timeWith_ms(frameTime_ms)

# Initialize smoothing variables
alpha = .1  # smoothing factor between 0 (slow) and 1 (no smoothing)
#Initialize direction 
direction  = "None" 

#all_frames_withDistance = [] # List to store all frames
#i = 0

good = False
while not good:
    success, raw_frame = videoOpbject.read() # Returns a boolean and the next frame
    if not success: # If the frame was not read successfully, break the loop
        print("Failed to read frame")
        exit()
    good, result, adjusted_time_ms = isPersonInFrame(raw_frame, start_frame) #newDim_Frame Checks if there is a person
    start_frame = start_frame +1


#Read through the specified frame count
for frame_Index in range(start_frame, end_frame): 
#for i in range(clipRunFrames):
    #frame_timestamp_ms = int((start_frame + i) * frameTime_ms)
    #i = i + 1 # Increment the frame index
    success, raw_frame = videoOpbject.read() # Returns a boolean and the next frame
    if not success: # If the frame was not read successfully, break the loop
        print("Failed to read frame")
        exit()
    newDim_Frame = raw_frame[min_height:max_height,min_width:max_width,:].copy() #Taking a full sized frame and
    #Shrinking it down to dimensions
    #Changes dimensions before finding landmarks
    #drawLandmark_square(raw_frame, min_width, max_width, min_height, max_height) #Returns a box around the person
    if newDim_Frame is not None: #Failsafe 
        good = False
        good, result, adjusted_time_ms = isPersonInFrame(newDim_Frame, frame_Index) #newDim_Frame Checks if there is a person
        #result is the landmarks
        if good and result is not None:
            #final_frame = blur_person_fullFrame(raw_frame, newDim_Frame, result, min_height, max_height, min_width, max_width)
            #final_frameCrop = blur_person_cropFrame(newDim_Frame, result)
            landmarks = result.pose_landmarks[0]
            #cropped_landLeft = landmarks[29]  # Left heel
            #cropped_landRight = landmarks[30]  # Right heel
            drawLandmark_circle(newDim_Frame, landmarks[29], [0,255,255],5) # Draw green landmarks before transition
            #drawLandmark_circle(final_frameCrop, landmarks[30], [255,0,0],5) # Draw green landmarks before transition
            landmarks_of_fullscreen(landmarks, min_width, max_width, min_height, max_height)
            #circle_landmarks = [15,16,30,29]  # shoulders, hips, knees, etc.
            #line_landmarks = [16,12,15,11,12,24,24,28,11,23,23,27,12,11,24,23,10,9]
            #for i in circle_landmarks:
            #    drawLandmark_circle(final_frame,landmarks[i],[0,255,0],5)
            #for i in range(0, len(line_landmarks), 2):
            #    drawLandmark_line(final_frame, landmarks[line_landmarks[i]], landmarks[line_landmarks[i + 1]], [0, 255, 0])
            #drawLandmark_circle(final_frame,landmarks[0],[0,255,0],15)
            min_width, max_width, min_height, max_height, maintain_dim  = crop_to_square(raw_frame, landmarks, direction ,maintain_dim) 
            smoothed_dim, min_width, max_width, min_height, max_height  = smooth_crop_dim(smoothed_dim, min_width, max_width, min_height, max_height) 
            resizedCropFrame = cv2.resize(newDim_Frame, squareDisplayRez) # Resize the frame for displayd 
            cv2.imshow("Frame to send model", resizedCropFrame) #displays frame
            resizedFrame = cv2.resize(raw_frame, displayRez) # Resize the frame for displayd
            cv2.imshow("Frame", resizedFrame) #displays frame
            
            # === Heel Y values (normalized and pixel)
            left_heel_y_norm = landmarks[29].y
            right_heel_y_norm = landmarks[30].y
            left_heel_y_px = left_heel_y_norm * height
            right_heel_y_px = right_heel_y_norm * height
            # === Distances using your function
            left_distHeel = find_dist_from_y(left_heel_y_px)
            right_distHeel = find_dist_from_y(right_heel_y_px)
        
             # === Toe Y values (normalized and pixel)
            left_toe_y_norm = landmarks[31].y
            right_toe_y_norm = landmarks[32].y
            left_toe_y_px = left_toe_y_norm * height
            right_toe_y_px = right_toe_y_norm * height
            # === Distances using your function
            left_distToe = find_dist_from_y(left_toe_y_px)
            right_distToe = find_dist_from_y(right_toe_y_px)
            total_seconds, timestamp = seconds_sinceMidnight(time_tracker, raw_frame, (frame_Index))
            #all_frames_withDistance.append({
            #"frame": newDim_Frame,
            #"right_heel": right_distHeel,
            #"left_heel": left_distHeel
            #})
            #newDim_H, newDim_W = newDim_Frame.shape[:2]
            #arrFrame = all_frames_withDistance[i]

            #cv2.putText(arrFrame["frame"], str(arrFrame["left_heel"]), (round(cropped_landLeft.x * newDim_W), round(cropped_landLeft.y * newDim_H)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #i = i + 1 # Increment the frame index
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                #frame_Index,
                timestamp,
                total_seconds,  # Convert to seconds
                left_distHeel,
                right_distHeel,
                left_distToe,
                right_distToe
            ])
            
    # === Save to CSV (✅ append only)
    # === Save to CSV (✅ append only)
#            if clip_start <= frame_Index < clip_end:
#                if out_full is None:
#                    out_full = cv2.VideoWriter(os.path.join(output_dir, "full_frame_clip.avi"), fourcc, fps, (width, height))

#                out_full.write(final_frame)
        else:
            if frame_Index % 2 ==0:
                min_width, max_width, min_height, max_height, direction = crop_to_Southhall() #, landmarks
            else:
                min_width, max_width, min_height, max_height, direction = crop_to_Northhall() #, landmarks
#            resizedCropFrame = cv2.resize(newDim_Frame, squareDisplayRez)
#            cv2.imshow("Frame to send model", resizedCropFrame)
#            resizedFrame = cv2.resize(raw_frame, displayRez) # Resize the frame for displayd
#            cv2.imshow("Frame", resizedFrame) #displays frame
  
        key1 = cv2.waitKey(0) # Wait for a key press
        if key1 == ord('q') & 0xFF: exit()
"""
frame_index = 0
play_mode = False  # False = manual step-through, True = autoplay
frame_delay = 30   # ms between frames in play mode

while 0 <= frame_index < len(all_frames_withDistance):

    # === Overlay info (frame number and time)
#    timestamp = str(datetime.timedelta(seconds=int(frame_index / videoOpbject.get(cv2.CAP_PROP_FPS))))
#    cv2.putText(frame, f"Frame: {frame_index}/{len(all_frames)-1}", (10, 30),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#    cv2.putText(frame, f"Time: {timestamp}", (10, 70),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # === Your processing function could go here
    # Example: landmarks, blurring, cropping, etc.
    entry = all_frames_withDistance[frame_index]
    #resizedframe = cv2.resize(entry["frame"], displayRez)
    cv2.imshow("Video Playback", entry["frame"])  # Display the current frame
    print(f"Left heel distance: {entry["left_heel"]}, Right heel distance: {entry["left_heel"]}")
    if play_mode:
        key = cv2.waitKey(frame_delay) & 0xFF
    else:
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely in manual mode

    if key == ord('q'):
        break
    elif key == 77 or key == ord('d'):  # → or 'd'
        frame_index += 1
    elif key == 81 or key == ord('g'):  # ← or 'a'
        frame_index -= 1
    elif key == ord('p'):  # Toggle play/pause
        play_mode = not play_mode
    elif key == ord('r'):  # Rewind to beginning
        frame_index = 0
    elif key == ord('f'):  # Jump forward 10 frames
        frame_index = min(frame_index + 10, len(all_frames_withDistance) - 1)
    elif key == ord('b'):  # Jump back 10 frames
        frame_index = max(frame_index - 10, 0)
    else:
        if play_mode:
            frame_index += 1  # Autoplay advances automatically

cv2.destroyAllWindows()

# Release writers after loop
#if out_full:
#     out_full.release()
#if out_crop:
#    out_crop.release()
"""
