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
import pytesseract # pip install pytesseract
import matplotlib as plt # matplotlib

# Media Pipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#export LIBGL_ALWAYS_SOFTWARE=1

## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
model_path = r"C:\Users\smitt\STARS\pose_landmarker_lite.task" # 5.5 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_full.task' # 9.0 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_heavy.task' # 29.2 MiB

#Video File
dir = r"E:\STARS\day1_data"
file = r"25_06_03_s1_1.asf"
fileName = f"{dir}/{file}"

## Open our video File
# Make a video object
videoOpbject = cv2.VideoCapture(fileName)

def getDateTime(frame):
    dateTime_img = frame[0:46,0:384] # Get the date time image from the top left corner
    dateTime_img_bw = cvtColor = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    dateTime_img_bw = 255 - dateTime_img_bw # Invert the image
    dateTime_outPut =pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    timeStr_num = 5
    dateTime_img = frame[0:46,0:400]# Use pytesseract to read the date time
    print(f"Time: {dateTime_outPut['text'][timeStr_num]} | conf: {dateTime_outPut['conf'][timeStr_num]}")

def drawLandmark_feet(frame, landmark):
    radius = 15
    thickness = 5
    color = [255, 0, 0] # Green for right foot
    center = int(landmark.x*w), int(landmark.y*h)
    print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")
    cv2.circle(frame, center, radius, color, thickness) # Draw a circle at the landmark position

def drawLandmark_line(frame, feet, hips):
    color = [255, 0, 0] # Red for left foot
    pt1_ft = (int(feet.x*w),int(feet.y*h)) 
    pt2_hips = (int(hips.x*w), int(hips.y*h))
    thickness = 5
    cv2.line(frame,pt1_ft,pt2_hips, color, thickness) # Draw a circle at the landmark position

if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()
fps = videoOpbject.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT)
w = int(videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

frame_timestamp_ms = 0.0
clipStartTime_s = 50.0
clipEndTime_s = 80.0
clipStartFrame = int(clipStartTime_s*fps)
clipEndFrame = int(clipEndTime_s*fps)

#videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame)) # Initial start point
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame))
for num in range(clipStartFrame, clipEndFrame): # Go through each frame
    frame_timestamp_ms += int(num * frameTime_ms)  # frame number to milliseconds
    sucess, frame = videoOpbject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    # Check to see if we actualy loaded a frame 
    if not sucess:
        print(f"Frame read failure")
        exit()
    #getDateTime(frame) # Get the date time from the frame

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))
    #annotated_image = draw_landmarks_on_image(image.numpy_view(), pose_landmarker_result)
    if len(pose_landmarker_result.pose_world_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        drawLandmark_feet(frame, landmarks[29]) # Draw the left heel
        drawLandmark_line(frame, landmarks[29],landmarks[23]) # Draw the left hip
        drawLandmark_line(frame, landmarks[30], landmarks[24]) # Draw the right hip
        drawLandmark_feet(frame, landmarks[30]) # Draw the right heel
        #print(f"L-H: {pose_landmarker_result.pose_world_landmarks[0][29].y}"
             # f"R-H: {pose_landmarker_result.pose_world_landmarks[0][30].y}") 

    #print(pose_landmarker_result)
    #Show the frame 

    rthp_x = int(landmarks[24].x * w)
    lfthp_x = int(landmarks[23].x * w)
    rtft_y = int(landmarks[32].y * h)
    #print(f"Right foot x: {rtft_x}, y: {rtft_y}")

    new_frame = frame[0:rtft_y + 50, lfthp_x - 100 :rthp_x + 100]
    print(f"The width is: {lfthp_x -rthp_x}")

    if new_frame.size > 0:
        cv2.imshow("Input", new_frame)
    else:
        print("Invalid crop — skipping frame")
    
    #frame = cv2.resize(new_frame, displayRez) # Resize the frame for display
    #qnew_frame = frame[w-int(landmarks[31].x)*w:h, int(h-landmarks[31].y)*h:h]
    #cv2.imshow("Input", frame)

    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()

  