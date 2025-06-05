###
# Loading a Video File
###
# Joshua Mehlman
# STARS, Summer 2025
###

import cv2  # pip install opencv-python

#Where is the file
videoDir = 'StudentData/25_06_03/'
subject = 'Subject_1'
#videoFile = 's3_B8A44FC4B25F_6-3-2025_4-08-45 PM.asf'
videoFile = '25_06_03_s1_1.asf'
fileDir = f"{videoDir}/{subject}/{videoFile}"

video = cv2.VideoCapture(fileDir)  # Load the video into cv2

# Get some information:
nFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  #how many frames?
width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
fps = video.get(cv2.CAP_PROP_FPS)  # float `height`
print(f"File: {videoFile} is {nFrames} frames")
print(f"W: {width}, H: {height}, FPS: {fps}")

for i in range(nFrames):
    ret, frame = video.read()
    print(f"Loading Frame: {i}")

    # Do something with what we got
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow(videoFile, frame)  # Show the frame
    key =  cv2.waitKey(0) 
    if key == ord('q'): break

video.release()