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
    frame[loc_y:loc_y+h, loc_x:loc_x+w] = overlay
    #print(f"c) Frame shape: {frame.shape}")

    return frame