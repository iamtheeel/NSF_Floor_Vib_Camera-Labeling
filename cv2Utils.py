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
from vibDataChunker import vibDataWindow
from velocity import calculate_avg_landMark_velocity 
from OCR_Detect import timeWith_ms
from distance_position import find_dist_from_y
import importlib

CHANNELS = {1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8x", 9:"8y", 10:"8z", 11:"9x", 12:"9y", 13:"9z", 14:"10", 15:"11", 16:"12", 17:"13", 18:"14", 19:"15", 20:"16"}

# Constants for Short-Time Fourier Transform (STFT)
# use exponent with base 2
FRAME_SIZE = 2**6
HOP_SIZE = 2**5 

RMSE_THRESHOLD = 0.001  # another layer of filtering

width = 2688
height = 1512

#From the chatbot:
def overlay_image(frame, overlay, loc_x, loc_y, dim_x, dim_y):
    # convert to cv2
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR) # Reorder the channels from RGBA to BGR
    overlay = cv2.resize(overlay, (dim_x, dim_y), interpolation=cv2.INTER_AREA) # Resize

    h, w = overlay.shape[:2]
    frame[loc_y-h:loc_y, loc_x:loc_x+w] = overlay

    return frame

def get_trigger_times(hdf5_path):
    """
    Extracts the triggerTime values (UNIX timestamps) from the HDF5 file.

    Parameters:
    - hdf5_path (str): Full path to the HDF5 file.

    Returns:
    - List of float or float: List of trigger times or 0.0 if not found.
    """
    trigger_times = []
    with h5py.File(hdf5_path, "r") as f:
        specific_params = f["experiment/specific_parameters"][:]
        for row in specific_params:
            param = row["parameter"].decode() if isinstance(row["parameter"], bytes) else row["parameter"]
            if param == "triggerTime":
                val = float(row["value"]) if not isinstance(row["value"], bytes) else float(row["value"].decode())
                trigger_times.append(val)
    return trigger_times if trigger_times else 0.0

def get_sample_rate(hdf5_path):
    """
    Extracts the sample rate from the HDF5 file.

    Parameters:
    - hdf5_path (str): Full path to the HDF5 file.

    Returns:
    - int: Sample rate or 0 if not found.
    """
    with h5py.File(hdf5_path, "r") as f:
        general_params = f["experiment/general_parameters"][:]
        fs_param = next(p for p in general_params if p["parameter"] == b"fs")
    return fs_param["value"]

def get_footstep_times(hdf5_path):
    CORRECTION = 1752130800
    with h5py.File(hdf5_path, "r") as f:
        data = f["experiment/data"][:]

    SR = get_sample_rate(hdf5_path)
    trigger_times = get_trigger_times(hdf5_path)

    df_events = []
    rise_all = []
    fall_all = []

    for sensor_index in range(20):
        channel_label = CHANNELS[sensor_index + 1]
        sensor_data = data[0, sensor_index, :]

        frequencies, times, Zxx = stft(sensor_data, fs=SR, nperseg=FRAME_SIZE, noverlap=FRAME_SIZE - HOP_SIZE)
        Y_scale = np.abs(Zxx) ** 2
        freq_bin_index = np.argmin(np.abs(frequencies - 100))
        power_100Hz = Y_scale[freq_bin_index, :]

        threshold = power_100Hz.mean() + 2 * power_100Hz.std()
        mask = power_100Hz > threshold

        rising_edges = np.where((mask[1:] & ~mask[:-1]))[0] + 1
        falling_edges = np.where((~mask[1:] & mask[:-1]))[0] + 1

        rising_times = times[rising_edges]
        falling_times = times[falling_edges]

        # Save all rising/falling events
        rise_all.extend([{"Sensor": channel_label, "Time (s)": t} for t in rising_times])
        fall_all.extend([{"Sensor": channel_label, "Time (s)": t} for t in falling_times])

        # Match rises and falls
        i, j = 0, 0
        while i < len(rising_times) and j < len(falling_times):
            if rising_times[i] < falling_times[j]:
                start_time = rising_times[i]
                end_time = falling_times[j]

                start_sample = int(start_time * SR)
                end_sample = int(end_time * SR)

                if end_sample > start_sample and end_sample <= len(sensor_data):
                    segment = sensor_data[start_sample:end_sample]
                    rms_energy = np.sqrt(np.mean(segment ** 2))
                else:
                    rms_energy = np.nan

                if rms_energy > RMSE_THRESHOLD:
                    df_events.append({
                        "Sensor": channel_label,
                        "Start Time (s)": start_time + trigger_times[0] - CORRECTION,
                        "End Time (s)": end_time + trigger_times[0] - CORRECTION,
                        "RMS Energy": rms_energy
                    })

                i += 1
                j += 1
            else:
                j += 1

    return pd.DataFrame(df_events)

def getDateTime(frame):
    dateTime_img = frame[0:46, 0:384, :]
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)
    dateTime_img_bw = 255 - dateTime_img_bw
    data = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    try:
        time_str = data['text'][5]
        AM_PM = data['text'][6]
        return f"{time_str}.{AM_PM}"
    except:
        return "OCR Error"

def seconds_sinceMidnight(timeWith_ms_class, raw_frame, frame_index):
    timestamp = getDateTime(raw_frame)
    if timestamp == "OCR Error":
        return None
    try:
        HHMMSS, AM_PM = timestamp.split('.')
        timestamp_withms = timeWith_ms_class.calc_ms(HHMMSS, frame_index)
        hours, minutes, seconds = timestamp_withms.split(':')
        seconds, milliseconds = seconds.split('.')
        if AM_PM == "PM" and hours != "12":
            hours = str(int(hours) + 12)
        elif AM_PM == "AM" and hours == "12":
            hours = "00"
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        return total_seconds
    except:
        return None
    
def find_dist_from_y(y_pix_height, resolution = False, debug = False):
    distance_from_cam = 7916.1069 / (y_pix_height + 86.1396) - 1.0263
    if resolution:
        print(f"Resolution innacuracy: {(7916.1069/(y_pix_height+86.1396)**2)*100:.3f}cm/px")
    if debug:
        print(f"{distance_from_cam:.3f}m")
    return distance_from_cam

#use y pixel value as the value
def find_resolution_px_dist(y_pix):
    resolution = (f"{(7916.1069/(y_pix+86.1396)**2)*100:.3f}cm/px")
    return resolution

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

def isPersonInFrame(frame, frameIndex, frameTime_ms, landmarkerVideo):
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
    adjust_width = (max_height - min_height) // 2
    center_width = width // 2
    min_width = center_width - adjust_width
    max_width = center_width + adjust_width
    direction = "North"  # <- Add this line
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

def get_foot_data(video_path):
    videoObject = cv2.VideoCapture(video_path)
    fps = videoObject.get(cv2.CAP_PROP_FPS)
    frame_count = int(videoObject.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_time_ms = 1000 / fps

    # === Initialize MediaPipe ===
    model_path = r"../media-pipeModels/pose_landmarker_lite.task"
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False
    )
    landmarkerVideo = PoseLandmarker.create_from_options(options)

    # === Time tracker setup ===
    time_tracker = timeWith_ms(frame_time_ms)

    # === Extract Heel Data with OCR Corrected Time ===
    foot_data = []

    frame_index = 0
    frame_height = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while frame_index < frame_count:
        success, frame = videoObject.read()
        if not success:
            break

        # OCR-based timestamp (seconds since midnight)
        true_time = seconds_sinceMidnight(time_tracker, frame, frame_index)

        # Run pose model
        frame_time = int(frame_index * frame_time_ms)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarkerVideo.detect_for_video(mp_image, frame_time)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # Get pixel Y-values for landmarks
            y29 = lm[29].y * frame_height  # left heel
            y30 = lm[30].y * frame_height  # right heel
            y31 = lm[31].y * frame_height  # left toe
            y32 = lm[32].y * frame_height  # right toe

            # Calculate distances
            left_distHeel = find_dist_from_y(y29)
            right_distHeel = find_dist_from_y(y30)
            left_distToe = find_dist_from_y(y31)
            right_distToe = find_dist_from_y(y32)

            foot_data.append({
                "frame": frame_index,
                "time_s_OCR": true_time,
                "left_heel_x": lm[29].x,
                "left_heel_y": lm[29].y,
                "right_heel_x": lm[30].x,
                "right_heel_y": lm[30].y,
                "left_toe_y": lm[31].y,
                "right_toe_y": lm[32].y,
                "left_distHeel_m": left_distHeel,
                "right_distHeel_m": right_distHeel,
                "left_distToe_m": left_distToe,
                "right_distToe_m": right_distToe,
            })


        frame_index += 1

    videoObject.release()

    return pd.DataFrame(foot_data)

def append_avg_foot_positions(footstep_times, foot_locations):
    # Columns to average
    columns_to_avg = ["left_heel_x", "left_heel_y", 
                      "right_heel_x", "right_heel_y", 
                      "left_distHeel_m", "right_distHeel_m", 
                      "left_distToe_m", "right_distToe_m"]
    
    # Initialize columns with NaNs
    for col in columns_to_avg:
        footstep_times[f"avg_{col}"] = np.nan

    for idx, row in footstep_times.iterrows():
        start_t = row["Start Time (s)"]
        end_t = row["End Time (s)"]
        
        # Select matching rows in foot_locations
        mask = (foot_locations["time_s_OCR"] >= start_t) & (foot_locations["time_s_OCR"] <= end_t)
        window = foot_locations[mask]

        if not window.empty:
            for col in columns_to_avg:
                footstep_times.at[idx, f"avg_{col}"] = window[col].mean()

    return footstep_times

def find_edges(frame):
    WIDTH = frame.shape[1]
    HEIGHT = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # initialize edge tracking variables
    left_tracker = 0
    right_tracker = WIDTH

    # Initialize edge lines
    left_line = ((0, 0), (0, HEIGHT))
    right_line = ((WIDTH, 0), (WIDTH, HEIGHT))

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Define bounding box
    LEFT = 800
    RIGHT = 1900
    TOP = 100
    BOTTOM = 1000

    # iterate through the detected lines
    if lines is not None:
        for line in lines:
            x_1, y_1, x_2, y_2 = line[0]

            # Correct orientation: y1 is top, y2 is bottom
            if y_2 > y_1:
                x1, y1 = x_1, y_1
                x2, y2 = x_2, y_2
            else:
                x1, y1 = x_2, y_2
                x2, y2 = x_1, y_1

            # check all points inside bounding box
            if (x1 < LEFT or x2 < LEFT):
                continue
            if (x1 > RIGHT or x2 > RIGHT):
                continue
            if (y1 < TOP or y2 < TOP):
                continue
            if (y1 > BOTTOM or y2 > BOTTOM):
                continue
            if y2 != y1: # avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                m = (x2 - x1) / (y2 - y1)
                # find right edges
                if 0.6 < m < 0.7:
                    x_top = int(x1 + m * (0 - y1))
                    x_bottom = int(x1 + m * (HEIGHT - y1))
                    if x_bottom < right_tracker:
                        right_tracker = x_bottom
                        right_floor_edge = ((x_top, 0), (x_bottom, HEIGHT))

                # find left edges
                elif -.3 < m < -.2:
                    x_top = int(x1 + m * (0 - y1))
                    x_bottom = int(x1 + m * (HEIGHT - y1))
                    if x_bottom > left_tracker:
                        left_tracker = x_bottom
                        left_floor_edge = ((x_top, 0), (x_bottom, HEIGHT))

    return right_floor_edge, left_floor_edge

def x_on_line_at_y(x1, y1, x2, y2, y_target):
    if y2 == y1:
        return None  # horizontal line — can't evaluate
    m = (x2 - x1) / (y2 - y1)
    return x1 + m * (y_target - y1)


def meters_from_left_wall(x, y, left_line, right_line, real_width_meters=3.3528):
    x_left = x_on_line_at_y(*left_line[0], *left_line[1], y)
    x_right = x_on_line_at_y(*right_line[0], *right_line[1], y)

    if x_left is None or x_right is None or x_right == x_left:
        return None  # invalid line or vertical overlap

    width_px = x_right - x_left
    fraction = (x - x_left) / width_px
    meters = fraction * real_width_meters
    return meters

def video_launcher(DEVELOPER, start_time=0, end_time=None):
    """
    Launches the video player with the specified video.
    
    Parameters:
    - video_path (str): Path to the video file.
    """
    settings = importlib.import_module(f"settings.{DEVELOPER}_settings")
    video_path = os.path.join(settings.FOLDER_DIR, settings.VIDEO_FILENAME)
    videoObject = cv2.VideoCapture(video_path) #open the video file and make a video object
    if not videoObject.isOpened():
        print("Error: Could not open video.")
        exit()
    print(f"Opening video: {video_path}")

    # get first frame to find floor to wall edges
    success, first_frame = videoObject.read() # Returns a boolean and the first frame
    if not success: # If the frame was not read successfully, break the loop
        print("Failed to read first frame")
        exit()
    else:
        right_floor_edge, left_floor_edge = find_edges(first_frame)

    # Video properties    
    fps = 30 # Frames per second
    fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT) #Frame count
    width = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH)) # Width of the video frame
    height = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of the video frame
    frameTime_ms = 1000/fps #How long of a time does each frame cover
    # Fit to the display
    dispFact = 2
    displayRez = (int(width/dispFact), int(height/dispFact))
    displayRezsquare = (int(height/dispFact), int(height/dispFact)) 
    model_path = settings.MEDIAPIPE_PATH

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

    start_frame = int(start_time * fps) # Start frame for the clip
    if end_time is None:
        end_frame = int(fCount)
    else:
        end_frame = int(end_time * fps)
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


    # === Prompt for user
    print(f"Press f to pause the video then you will be able to use other keys to navigate through the video frames. Press q to quit.")

    # === Sets the video to specified index
    frame_Index = start_frame
    videoObject.set(cv2.CAP_PROP_POS_FRAMES, frame_Index)

    # === Begin process of cropping, saving, and playback
    waitKeyP = 1
    toeVel_mps = 0
    framewith_data = 0

    vibImage_rgba = None
    windowName = "Main Frame:"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    while frame_Index < end_frame:
        i = frame_Index - start_frame #index for track_frames array
        # === Reads and loads new frames in array
        if track_frames[i]['frame'] is None: 
            #print(f"frame_Index: {frame_Index}, i: {i}")
            success, raw_frame = videoObject.read() # Returns a boolean and the next frame
            if not success: # If the frame was not read successfully, break the loop
                print("Failed to read frame")
                exit()
            # === Saves seconds since midnight
            total_seconds = seconds_sinceMidnight(time_tracker, raw_frame, frame_Index)
            # === Crops full frame. Draws the cropped area on full frame
            newDim_Frame = raw_frame[min_height:max_height,min_width:max_width,:].copy() #crops frame
            cv2.rectangle(raw_frame, (min_width,max_height), (max_width, min_height), [255,0,0], 5)
            # ===Is there a cropped frame to send to model?
            if newDim_Frame is not None: 
                good = False
            # === Returns landmarks based on person
                good, result, adjusted_time_ms = isPersonInFrame(newDim_Frame, frame_Index, frameTime_ms, landmarkerVideo)

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
                    
                    track_frames[i]["seconds_sinceMid"] = total_seconds
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
                            # vibImage_rgba = vib.vib_get(time=total_seconds, distanceFromCam=50)
                            


                    if vibImage_rgba is not None:
                        raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba, loc_x=550, loc_y=1000, dim_x=300, dim_y=300) # overlay at this position


                    track_frames[i]["toeVel"] = toeVel_mps
                    track_frames[i]["heelVel"] = toeVel_mps

                    # Calculate horizontal position from left wall
                    left_x = landmarks[29].x * width
                    right_x = landmarks[30].x * width
                    y_pos = (landmarks[29].y + landmarks[30].y) / 2 * height  # average y for consistency

                    left_x_m = meters_from_left_wall(left_x, y_pos, left_floor_edge, right_floor_edge)
                    right_x_m = meters_from_left_wall(right_x, y_pos, left_floor_edge, right_floor_edge)

                    text = [
                        f"Left Toe: {track_frames[i]['LeftToe_Dist']:.2f} m", 
                        f"Left Heel: {track_frames[i]['LeftHeel_Dist']:.2f} m", 
                        f"Left X: {left_x_m:.2f} m" if left_x_m else "Left X: N/A",
                        f"Right Toe: {track_frames[i]['RightToe_Dist']:.2f} m",
                        f"Right Heel: {track_frames[i]['RightHeel_Dist']:.2f} m",
                        f"Right X: {right_x_m:.2f} m" if right_x_m else "Right X: N/A",
                        f"Toe Vel: {track_frames[i]['toeVel']:.2f} m/s",
                        f"Heel Vel: {track_frames[i]['heelVel']:.2f} m/s",
                        f"Seconds: {track_frames[i]['seconds_sinceMid']:.3f} s"
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
                # ===resize for viewing and save in array
                resized_rawframe = cv2.resize(raw_frame, displayRez)
                print(f"shape | raw_frame {raw_frame.shape}, resized_rawframe {resized_rawframe.shape}")

                resizedframe = cv2.resize(newDim_Frame, displayRezsquare)
                track_frames[i]["frame"] = resized_rawframe
                track_frames[i]["cropped_frame"] = resizedframe
                put_text(text, track_frames[i]["cropped_frame"])
                put_text(text, track_frames[i]["frame"])
        else:
            resized_rawframe = track_frames[i]["frame"]
            resizedframe = track_frames[i]["cropped_frame"]

        cv2.imshow("Zoomed Frame: ", resizedframe)
        cv2.imshow(windowName, resized_rawframe)
        cv2.resizeWindow(windowName, 1433, 756) #TODO: use from vars

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
            
###
# STARS
# MIC Lab
# Spring, 2025
###
# Image handeling utils
###

import cv2

#From the chatbot:
def overlay_image(frame, overlay, loc_x, loc_y, dim_x, dim_y):
    # convert to cv2
    print(f"a) Frame shape: {frame.shape}")
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR) # Reorder the channels from RGBA to BGR
    overlay = cv2.resize(overlay, (dim_x, dim_y), interpolation=cv2.INTER_AREA) # Resize

    h, w = overlay.shape[:2]
    #print(f"b) Frame shape: {frame.shape}")
    frame[loc_y-h:loc_y, loc_x:loc_x+w] = overlay
    #print(f"c) Frame shape: {frame.shape}")

    return frame