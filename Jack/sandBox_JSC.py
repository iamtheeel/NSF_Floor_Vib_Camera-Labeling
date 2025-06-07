import cv2 # pip install opencv-python
import time

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_06_03\Subject_2'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}/{file}" # Path to the video file
print(filename)

videoObject = cv2.VideoCapture(filename) # Open the video file
if not videoObject.isOpened():
    print("Error: Could not open video file.")
    exit()
print(f"Loaded: {filename}")

def nothingburger(fpsPeram, heightPeram, widthPeram, frameCountPeram):
    fps = videoObject.get(fpsPeram)
    height = videoObject.get(heightPeram)
    width = videoObject.get(widthPeram)
    frame_count = videoObject.get(frameCountPeram)
    return fps, height, width, frame_count

fps, height, width, frame_count = nothingburger(cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_COUNT)

print(f"FPS: {fps}, Height: {height}, Width: {width}, Frame Count: {frame_count}")
