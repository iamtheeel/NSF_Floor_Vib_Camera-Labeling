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

# Media Pipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#export LIBGL_ALWAYS_SOFTWARE=1

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_lite.task' # 5.5 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_full.task' # 9.0 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_heavy.task' # 29.2 MiB

#Video File
dir = 'StudentData/25_06_03/Subject_1'
file = '25_06_03_s1_1.asf'
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
frame_timestamp_ms = 0
for i in range(int(fCount)): # Go through each frame
    frame_timestamp_ms += int(i*frameTime_ms) # for i = 0, no increment, i=1 will go to next time
    sucess, frame = videoOpbject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    # Check to see if we actualy loaded a frame 
    if not sucess:
        print(f"Frame read failure")
        exit()

    #frame = cv2.resize(frame, displayRez)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to grey scale
    cv2.imshow("Input", frame)
    print(f"Frame: {i}, timeStamp: {frame_timestamp_ms}")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    

    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()
