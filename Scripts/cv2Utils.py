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


def overlay_image(frame, overlay, loc_x, loc_y, dim_x, dim_y):
    # convert to cv2
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR) # Reorder the channels from RGBA to BGR
    overlay = cv2.resize(overlay, (dim_x, dim_y), interpolation=cv2.INTER_AREA) # Resize

    h, w = overlay.shape[:2]
    frame[loc_y-h:loc_y, loc_x:loc_x+w] = overlay

    return frame