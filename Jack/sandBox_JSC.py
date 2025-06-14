####
#   Jack Capito
#   STARS Summer 2025
#   Dr J Lab
###
# mediapipe trials
# open our video file
####

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
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_lite.task" #5.5 MB
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_full.task" #9.0 MB
model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_heavy.task" #29.2 MB

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_06_11'
file = 'subject_2_test_3_6-11-2025_5-46-23 PM.asf'
fileName = f"{dir}/{file}"  # Path to the video file


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
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False  # <--- Set this to False
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
clipStartTime_s = 3
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

    #getDateTime(frame) #Read the date and time from the upper left of the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if len(pose_landmarker_result.pose_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        # Get image height for pixel calculation
        h, w, _ = frame.shape

        # Left heel (landmarks[29])
        left_heel_y_px = int((landmarks[29].y * h))
        #left_heel_dist = 7927.4586 / (left_heel_y_px + 85.6982) + -1.0522
        left_heel_dist = 7916.1069 / (left_heel_y_px + 86.1396) + -1.0263

        # Right heel (landmarks[30])
        right_heel_y_px = int((landmarks[30].y * h))
        #right_heel_dist = 7927.4586 / (right_heel_y_px + 85.6982) + -1.0522
        right_heel_dist = 7916.1069 / (right_heel_y_px + 86.1396) + -1.0263

        print(f"Left heel: y={left_heel_y_px}, distance={left_heel_dist:.2f} meters")
        print(f"Right heel: y={right_heel_y_px}, distance={right_heel_dist:.2f} meters")

        # Optionally, draw the distance on the frame
        cv2.putText(frame, f"L: {left_heel_dist:.2f}", (int(landmarks[29].x * w), left_heel_y_px - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"R: {right_heel_dist:.2f}", (int(landmarks[30].x * w), right_heel_y_px - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(frame, (int(landmarks[29].x * w), left_heel_y_px), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(landmarks[30].x * w), right_heel_y_px), 3, (0, 255, 0), -1)

        landmarks_w = pose_landmarker_result.pose_world_landmarks[0] 
        dX = landmarks_w[29].x - landmarks_w[30].x
        dY = landmarks_w[29].y - landmarks_w[30].y
        dZ = landmarks_w[29].z - landmarks_w[30].z
        strideLen = math.sqrt(math.pow(dX,2) + math.pow(dY, 2) + math.pow(dZ, 2))
        print(f"stride length: {strideLen:.3f}m, {strideLen*3.28084:.3f} ft")
    if pose_landmarker_result.segmentation_masks is not None:
        mask = pose_landmarker_result.segmentation_masks[0].numpy_view()
        cv2.imshow("Seg mask", mask)

    #Show the frame 
    #draw circle, or line, or rectange.
    #cv2.circle(frame, (200, 100), 20, [0, 0, 100], 1)
    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)
    #print(f"Frame: {i}, timeStamp: {frame_timestamp_ms} ms")


    key = cv2.waitKey(int(0))
    if key == ord('q') & 0xFF: exit()
