###
# STARS
# MIC Lab
# Spring, 2025
###
# Image handling utils
###

import h5py
import numpy as np
import pandas as pd
from scipy.signal import stft
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pytesseract

import importlib



from Scripts.velocity import calculate_avg_landMark_velocity 
from Scripts.OCR_Detect import timeWith_ms
from Scripts.distance_position import find_dist_from_y
from Scripts.vibDataChunker import vibDataWindow



CHANNELS = {1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8x", 9:"8y", 10:"8z", 11:"9x", 12:"9y", 13:"9z", 14:"10", 15:"11", 16:"12", 17:"13", 18:"14", 19:"15", 20:"16"}

# Constants for Short-Time Fourier Transform (STFT)
# use exponent with base 2
FRAME_SIZE = 2**6
HOP_SIZE = 2**5 

RMSE_THRESHOLD = 0.001  # another layer of filtering

width = 2688
height = 1512



def isPersonInFrame(frame, frameIndex, frameTime_ms, landmarkerVideo): #(frame, frameIndex)
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


def seconds_sinceMidnight(raw_frame, frame_Index, time_tracker):
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
    return round(min_width), round(max_width), round(min_height), round(max_height)#, direction

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
    #if result.segmentation_masks is None:
    #    return raw_frame

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

def side_bar_text(frame, track_frames, pridictied_location):

    text_array = [
        f"Seconds: {track_frames['seconds_sinceMid']:.3f} s",
        f"Left Heel: {track_frames['LeftHeel_Dist']:.2f} m",
        f"Left Toe: {track_frames['LeftToe_Dist']:.2f} m",
        f"Right Heel: {track_frames['RightHeel_Dist']:.2f} m",
        f"Right Heel Location: X: {pridictied_location[0][0]/100:.2f}m Y: {pridictied_location[0][1]/100:.2f}",
        f"Difference in Y Predictions: {abs(pridictied_location[0][1]/100 - track_frames['RightHeel_Dist']):.2f}m",
        f"Right Toe: {track_frames['RightToe_Dist']:.2f} m",
        'Previous Window:',
        f"Toe Vel: {track_frames['toeVel']:.2f} m/s",
        f"Heel Vel: {track_frames['heelVel']:.2f} m/s"
        
    ]
    
    put_text(text_array, frame)

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




def overlay_image(frame, overlay, loc_x, loc_y, dim_x, dim_y):
    # convert to cv2
    if overlay is None or not hasattr(overlay, "size") or overlay.size == 0:
        return frame
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR) # Reorder the channels from RGBA to BGR
    overlay = cv2.resize(overlay, (dim_x, dim_y), interpolation=cv2.INTER_AREA) # Resize

    h, w = overlay.shape[:2]
    frame[loc_y-h:loc_y, loc_x:loc_x+w] = overlay

    return frame

def draw_box(frame, upper_left, bottom_right, color=(0, 255, 0), thickness=2):
    """Draws a bounding box on the frame."""
    cv2.rectangle(frame, upper_left, bottom_right, color, thickness)


def draw_feet_points(frame, landmarks):
    drawLandmark_circle(frame, landmarks[31], [230, 216, 173], 5)
    drawLandmark_circle(frame, landmarks[29], [139, 0, 0], 5)
    drawLandmark_circle(frame, landmarks[32], [102, 102, 255], 5)
    drawLandmark_circle(frame, landmarks[30], [0, 0, 139], 5)

def draw_vibration_overlay(frame, vibImage_rgba):

    # location of the vibration overlays 
    locations = [
        (950, 7.59),
        (1900, 10.26),
        (1000, 12.32),
        (1750, 13.99),
        (1050, 17),
        (1600, 23.32),
        (1100, 26.82)
    ]

    for idx, (loc_x, dist) in enumerate(locations):
        frame = overlay_image(
            frame.copy(),
            vibImage_rgba[idx],
            loc_x=loc_x,
            loc_y=int(findPixfromDist(dist)),
            dim_x=200,
            dim_y=200
        )

    return frame

