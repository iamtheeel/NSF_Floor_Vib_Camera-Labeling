####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Opens a video file in OpenCV
####

import cv2 # pip install opencv-python

dir = 'StudentData/25_06_03/Subject_1'
file = '25_06_03_s1_1.asf'
fileName = f"{dir}/{file}"

videoOpbject = cv2.VideoCapture(fileName)

if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

fps = videoOpbject.get(cv2.CAP_PROP_FPS)
print(f"Loded: {fileName}, FramesPerS: {fps}hz")