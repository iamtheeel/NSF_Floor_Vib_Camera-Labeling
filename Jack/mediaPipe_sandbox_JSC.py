####
#   Jack Capito
#   STARS Summer 2025
#   Dr J Lab
###
# mediapipe trials
# open our video file
####

import time
import cv2
import mediapipe as mp  # pip install mediapipe
from mediapipe.tasks import python  # pip install mediapipe-tasks
from mediapipe.tasks.python import vision  # pip install mediapipe-tasks
import pytesseract  # pip install pytesseract

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_lite.task" #5.5 MB
#model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_full.task" #9.0 MB
model_path = r"C:\Users\notyo\Documents\STARS\mediapipe\pose_landmarker_heavy.task" #29.2 MB

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25-06-26'
file = 'wave_3_6-26-2025_12-03-18 PM.asf'
fileName = f"{dir}/{file}"  # Path to the video file

## Open our video File
# Make a video object
videoObject = cv2.VideoCapture(fileName)

if not videoObject.isOpened():
    print("Error: Could not open video.")
    exit()
fps = 30
fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT)
w = videoObject.get(cv2.CAP_PROP_FRAME_WIDTH)
h = videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
                               )

landmarker = PoseLandmarker.create_from_options(options)
#exit()

def getDateTime(frame):
    dateTime_img = frame[0:45, 0:384]  # Crop the date and time area
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    dateTime_img_bw = cv2.bitwise_not(dateTime_img_bw)  # Invert the colors
    dateTime_outPut = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    dateStr_num = 4 
    timeStr_num = 5
    print(f"Date: {dateTime_outPut['text'][dateStr_num]}, conf:{dateTime_outPut['conf'][dateStr_num]}",
          f"Time: {dateTime_outPut['text'][timeStr_num]}, conf:{dateTime_outPut['conf'][timeStr_num]}")


def getTime(frame):
    dateTime_img = frame[0:45, 0:384]  # Crop the date and time area
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    dateTime_img_bw = cv2.bitwise_not(dateTime_img_bw)  # Invert the colors
    dateTime_outPut = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    timeStr_num = 5
    return dateTime_outPut['text'][timeStr_num]


def draw_landmarks_on_image(image, pose_landmarker_result):
    if not pose_landmarker_result.pose_landmarks:
        return image
    h, w, _ = image.shape
    for landmark in pose_landmarker_result.pose_landmarks[0]:
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        cv2.circle(image, (x_px, y_px), 4, (0, 255, 0), -1)  # Green dot
    return image

clipStartSecs = .1 #Start at this many seconds into the video
clipEndSecs = 50 #End this many seconds into the video

clipStartTime_f = fps * clipStartSecs
clipEndTime_f = ((fps * clipEndSecs) - int(clipStartTime_f))
if clipEndTime_f < 0:
    print(f"Clip is negative length, make sure clipEndSecs is greater than clipStartSecs")
    exit()


prev_time = None
frames_since_last = 0

    

videoObject.set(cv2.CAP_PROP_POS_FRAMES, clipStartTime_f)
print(fps)
print(1000/fps)
newFPS = 1000/fps
frame_timestamp_ms = 0 # Skip to frame x before starting the loop

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out_path = r'C:\Users\notyo\Documents\STARS\goutput.mp4'  # Change as needed
out = cv2.VideoWriter(out_path, fourcc, fps, displayRez)

for i in range(int(clipEndTime_f)): # Go through each frame
    frame_timestamp_ms += int(i*frameTime_ms) # for i = 0, no increment, i=1 will go to next time
    sucess, frame = videoObject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    # Check to see if we actualy loaded a frame 
    if not sucess:
        print(f"Frame read failure")
        exit()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    annotated_image = draw_landmarks_on_image(frame.copy(), pose_landmarker_result)
    #print(pose_landmarker_result)

    # Assume 'landmarks' is your list of pose landmarks and 'frame' is your image
    h, w, _ = frame.shape

    # MediaPipe Pose indices for foot landmarks
    my_list = range(33)

    if pose_landmarker_result.pose_landmarks:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        for idx in my_list:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    ##time rollover variables
    
    current_time = getTime(frame)

    if prev_time is None:
        prev_time = current_time

    if current_time != prev_time:
        print(f"Frames since last rollover: {frames_since_last}")
        frames_since_last = 0  # Start counting for the new time value
        prev_time = current_time

    frames_since_last += 1
    # Draw landmarks on the frame
    #frame_with_landmarks = draw_landmarks_on_image(frame.copy(), pose_landmarker_result)

    #Show the frame 
    frame = cv2.resize(frame, displayRez)
    out.write(frame)
    cv2.imshow("Input", frame)
    #print(f"Frame: {i}, timeStamp: {(i*newFPS - 12*newFPS)/1000}")


    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()
out.release()
