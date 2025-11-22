# draw_utils.py
# Utilities for drawing landmarks, circles, overlays, and text

import cv2
import numpy as np


def drawLandmark_circle(frame, landmark, color, radius):
    h, w, _ = frame.shape
    cx = int(landmark.x * w)
    cy = int(landmark.y * h)
    cv2.circle(frame, (cx, cy), radius, color, -1)


def put_text(lines, frame, x=50, y=50, scale=0.7, color=(255,255,255)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i*25), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def overlay_image(base_img, overlay_rgba, loc_x, loc_y, dim_x, dim_y):
    overlay_resized = cv2.resize(overlay_rgba, (dim_x, dim_y))

    # Extract BGR + Alpha
    if overlay_resized.shape[2] == 4:
        b, g, r, a = cv2.split(overlay_resized)
        mask = a / 255.0
    else:
        b, g, r = cv2.split(overlay_resized)
        mask = np.ones((dim_y, dim_x))

    overlay_rgb = cv2.merge([b, g, r])

    # Region of interest
    roi = base_img[loc_y:loc_y+dim_y, loc_x:loc_x+dim_x]

    # Blend
    blended = roi * (1 - mask[..., None]) + overlay_rgb * (mask[..., None])
    base_img[loc_y:loc_y+dim_y, loc_x:loc_x+dim_x] = blended.astype(np.uint8)
    return base_img
