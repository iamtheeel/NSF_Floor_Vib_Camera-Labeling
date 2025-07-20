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

CHANNELS = {1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8x", 9:"8y", 10:"8z", 11:"9x", 12:"9y", 13:"9z", 14:"10", 15:"11", 16:"12", 17:"13", 18:"14", 19:"15", 20:"16"}

# Constants for Short-Time Fourier Transform (STFT)
# use exponent with base 2
FRAME_SIZE = 2**6
HOP_SIZE = 2**5 

RMSE_THRESHOLD = 0.001  # another layer of filtering

def overlay_image(bg, overlay, loc_x, loc_y, dim_x, dim_y):
    img_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR) # convert to cv2
    resized_img = cv2.resize(img_bgr, (dim_x, dim_y), interpolation=cv2.INTER_AREA) # Resize

    h, w = resized_img.shape[:2]
    bg[loc_y:loc_y+h, loc_x:loc_x+w] = resized_img
    return bg

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