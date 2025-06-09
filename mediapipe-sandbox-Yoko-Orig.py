###
# Loading a Video File
###
# Yoko Lu
# SFSU STARS, Summer 2025
###
#Playing with MediaPipe
#Open our video file
####

#Built ins
import time

#d
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Configuration
#Mediapipe model
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task" #9.4MB
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task" #30.7MB
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task" #5.8B

#Video File
dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}{file}"  # Full path to the video file

#From https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#video ###
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
                                base_options=BaseOptions(model_asset_path=model_path),
                                running_mode=VisionRunningMode.VIDEO
            )

with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    


