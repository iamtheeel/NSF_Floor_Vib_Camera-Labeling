###
# STARS
# MIC Lab
# Spring, 2025
###
# Image handeling utils
###

import cv2

#From the chatbot:
def overlay_image(bg, overlay, loc_x, loc_y, dim_x, dim_y):
    img_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR) # convert to cv2
    resized_img = cv2.resize(img_bgr, (dim_x, dim_y), interpolation=cv2.INTER_AREA) # Resize

    h, w = resized_img.shape[:2]
    bg[loc_y:loc_y+h, loc_x:loc_x+w] = resized_img
    return bg