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
import math

#Third party
import cv2 # pip install opencv-python
import numpy as np

# Media Pipe
import mediapipe as mp  #pip install mediapipe
from mediapipe.tasks import python   # installs with mediapipe
from mediapipe.tasks.python import vision   # installs with mediapipe

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_lite.task' # 5.5 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_full.task' # 9.0 MiB
model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_heavy.task' # 29.2 MiB

#Video File
dir = 'StudentData/25_06_03/Subject_2'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
#dir = 'StudentData/25_06_09'  
#file = 'B8A44FC4B25F_6-9-2025_11-54-27 AM.asf' #H.264, GOP = 150
#file  ='B8A44FC4B25F_6-9-2025_12-13-11 PM.a' #H.264, GOP = 1
#file = 'output_allkey.mp4'
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

frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
dispFact = 2
displayRez = (int(w/dispFact), int(h/dispFact))



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
                                output_segmentation_masks=True
                               )
landmarker = PoseLandmarker.create_from_options(options)

import pytesseract #pip install pytesseract to do OCR/ICR
def getDateTime(frame):
    dateTime_img = frame[0:46, 0:384, :] # Get just the time
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY) # Convert to grey scale
    dateTime_img_bw = 255 - dateTime_img_bw #Invert the image
    #print(f"dateTime_img type: {type(dateTime_img)}, shape: {dateTime_img.shape}")
    #print(dateTime_img[0:30, 25:40])
    dateTime_outPut = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    timeStr_num = 5
    print(f"Time: {dateTime_outPut['text'][timeStr_num]} | conf: {dateTime_outPut['conf'][timeStr_num]}%") 

def drawLandmark(frame, landmark, color):
    radius = 15
    thickness = 5 # filled in 
    #print(f"x: {landmark.x}, y: {landmark.y}")  
    center = [int(landmark.x*w), int(landmark.y*h)]
    cv2.circle(frame, center , radius, color, thickness)

#exit()
clipRunTime_s = 0
clipStartTime_s = 30
clipEndTime_s = clipStartTime_s + clipRunTime_s
clipStartFrame = clipStartTime_s*fps
if clipRunTime_s == 0:
    clipRunFrames = int(fCount - clipStartFrame)
else:
    clipRunFrames = int((clipEndTime_s- clipStartTime_s)*fps)

#videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, clipStartFrame) # Initial start point, goes to first keyframe
videoOpbject.set(cv2.CAP_PROP_POS_MSEC, clipStartTime_s*1000) # Initial start point
#whichFrame = videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)

frame_timestamp_ms = 0
for i in range(clipRunFrames): # Go through each frame this many times
    frame_timestamp_ms += int(i*frameTime_ms) # for i = 0, no increment, i=1 will go to next time
    sucess, frame = videoOpbject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    # Check to see if we actualy loaded a frame 
    if not sucess:
        print(f"Frame read failure")
        exit()

    getDateTime(frame) #Read the date and time from the upper left of the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if len(pose_landmarker_result.pose_landmarks)  > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0] 
        drawLandmark(frame, landmarks[0], [0, 0, 0]) # Nose
        drawLandmark(frame, landmarks[29], [255, 0, 0]) # Left heel
        drawLandmark(frame, landmarks[30], [0, 255, 0]) # Right heel

        landmarks_w = pose_landmarker_result.pose_world_landmarks[0] 
        dX = landmarks_w[29].x - landmarks_w[30].x
        dY = landmarks_w[29].y - landmarks_w[30].y
        dZ = landmarks_w[29].z - landmarks_w[30].z
        strideLen = math.sqrt(math.pow(dX,2) + math.pow(dY, 2) + math.pow(dZ, 2))
        print(f"stride length: {strideLen:.3f}m, {strideLen*3.28084:.3f} ft")
    if pose_landmarker_result.segmentation_masks is not None:
        #mask = np.array(pose_landmarker_result.segmentation_masks[0])
        mask = pose_landmarker_result.segmentation_masks[0].numpy_view()
        #mask = (mask * 255).astype(np.uint8) # was 0.0 - 1.0 --> 0 - 255
        cv2.imshow("Seg mask", mask)

    #Show the frame 
    #draw circle, or line, or rectange.
    #cv2.circle(frame, (200, 100), 20, [0, 0, 100], 1)
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)
    #print(f"Frame: {i}, timeStamp: {frame_timestamp_ms} ms")


    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()
