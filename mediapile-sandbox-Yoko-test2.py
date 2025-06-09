####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Playing with media pipe
# Open our video file
####

## Imports
#Built ins
import time

#Third party
import cv2 # pip install opencv-python
import pytesseract

# Media Pipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#export LIBGL_ALWAYS_SOFTWARE=1

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task" #9.4MB
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task" #30.7MB
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task" #5.8B

#Video File
dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{dir}/{file}"

## Open our video File
# Make a video object
videoOpbject = cv2.VideoCapture(fileName)

if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()
fps = videoOpbject.get(cv2.CAP_PROP_FPS)
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT)
w = videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH)
h = videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT)

frameTime_ms = 1000/fps
# Display
dispFact = 2
displayRez = (256, 256)



### From https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#video ###
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
                                base_options=BaseOptions(model_asset_path=model_path,
                                                         delegate=BaseOptions.Delegate.CPU # Default is GPU, and I anin't got none
                                                         ),
                                running_mode=VisionRunningMode.VIDEO,
                               )

landmarker = PoseLandmarker.create_from_options(options)
#with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
#exit()

#This code below runs the video for 10 seconds, then stops.
clipStartTime_s = 10
clipEndTime_s = 20
clipRunTime_s = clipEndTime_s - clipStartTime_s
clipStartFrame = int(clipStartTime_s * fps)
clipRunFrames = int((clipEndTime_s - clipStartTime_s) * fps)

videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, clipStartFrame) #at 10 seconds
frame_timestamp_ms = int(clipStartFrame * frameTime_ms)
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, clipStartFrame_s * 1000)
frame_timestamp_ms = int(clipStartFrame * frameTime_ms)

success, frame = videoOpbject.read()

current_time_ms = videoOpbject.get(cv2.CAP_PROP_POS_MSEC)
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, clipStartFrame) # Set the video to the frame we want to read
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, clipEndTime_s * 1000) # Set the video to the end time fram
whichFrame = videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)

print(f"Time: {current_time_ms / 1000} sec, Frame: {int(clipStartTime_s)}")

frame_timestamp_ms = 0

for i in range(clipRunFrames):
    success, frame = videoOpbject.read()
    if not success:
        print(f"❌ Frame read failure at frame {i}")
        break
    frame_timestamp_ms += int(frameTime_ms)
    # Process the frame here (e.g., pose detection, save, etc.)


    #frame = cv2.resize(frame, displayRez)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to grey scale
    cv2.imshow("Input", frame)
    print(f"Frame: {i}, timeStamp: {frame_timestamp_ms}")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

def getDateTime(frame):



    getDateTime(frame)

    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()
