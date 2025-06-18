####
#   Kara-Leah Smittle
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
import cv2 # opencv-python
import pytesseract # pytesseract
import matplotlib as plt # matplotlib
import numpy as np # numpy

# Media Pipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


## Configureations:
# Media pipe model: 
#Pose detector: 224 x 224 x 3
#Pose landmarker: 256 x 256 x 3 
model_path = r"C:\Users\smitt\STARS\pose_landmarker_lite.task" # 5.5 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_full.task' # 9.0 MiB
#model_path = '/home/josh/Documents/MIC/shake/STARS/media-pipeModels/pose_landmarker_heavy.task' # 29.2 MiB


#Video File
dir = r"E:\STARS\day1_data"
file = r"Yoko_6_3_2025_4_08_45.asf"
fileName = f"{dir}/{file}"

#Global variables
videoOpbject = cv2.VideoCapture(fileName) #open the video file and make a video object
if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties    
fps = videoOpbject.get(cv2.CAP_PROP_FPS) # Frames per second
#print(f"FPS: {fps}")
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT) #Frame count
height, width, _ = videoOpbject.read()[1].shape # Get the width and height of the video frame

frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
dispFact = 2
displayRez = (int(width/dispFact), int(height/dispFact))


#functions
def getDateTime(frame):
    dateTime_img = frame[0:46,0:384] # Get the date time image from the top left corner
    dateTime_img_bw = cvtColor = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    dateTime_img_bw = 255 - dateTime_img_bw # Invert the image
    dateTime_outPut =pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT) # Use pytesseract to read the date time
    timeStr_num = 5 # The number of the time string in the output dictionary
    dateTime_img = frame[0:46,0:400]# Crop the date time image from the frame
    print(f"Time: {dateTime_outPut['text'][timeStr_num]} | conf: {dateTime_outPut['conf'][timeStr_num]}") #output date time and confidence 

def drawLandmark_circle(frame, landmark):
    radius = 15
    thickness = 5
    color = [255, 0, 0] #Circle will be red
    center = int(landmark.x*width), int(landmark.y*height) #place center of the circle at the landmark position
    #print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")
    cv2.circle(frame, center, radius, color, thickness) # Draw a circle at the landmark position

def drawLandmark_line(frame, feet, hips):
    color = [255, 0, 0] # Line will be red
    pt1_ft = (int(feet.x*width),int(feet.y*height)) #First point is on the feet
    pt2_hips = (int(hips.x*width), int(hips.y*height)) #second point is on the hips
    thickness = 5
    cv2.line(frame,pt1_ft,pt2_hips, color, thickness) # Draw a line from the feet to the hips

def crop_with_padding(frame, lfthip, rthip, rtft_y, crop_width=256):
    # Calculate center between left and right hips
    hip_x = (lfthip + rthip) // 2 # Calculate the center x position between the left and right hips
    # Set crop bounds equidistant from the center
    x1 = hip_x - crop_width // 2
    x2 = hip_x + crop_width // 2
    # Calculate how much we are out of bounds
    pad_left   = max(0, -x1) #x1 is negative if it extends beyond the left edge of the frame
    pad_right  = max(0, x2 - width) #x1 is negative if it extends beyond the right edge of the frame
        #Take max of numbers to get padding amount
    # Apply padding if needed
    frame_padded = cv2.copyMakeBorder(frame,0,0,pad_left,pad_right,borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))  # Black padding
     # Update crop coordinates for the padded image
    x1_padded = x1 + pad_left
    x2_padded = x2 + pad_right
    #y1_padded = y1 + pad_top
    #y2_padded = y2 + pad_top
    # Crop the padded frame
    cropped = frame_padded[0: rtft_y + 50, x1_padded:x2_padded]
    print(f"Crop bounds: x1: {x1_padded}, x2: {x2_padded}, y1: 0, y2: {rtft_y + 50}")
    return cropped   #Return the cropped frame with padding



#mediaPipe settings
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
                                running_mode=VisionRunningMode.VIDEO, # Set the running mode to video
                               )

landmarker = PoseLandmarker.create_from_options(options)
#exit()

#frame_timestamp_ms = 0.0
#clipStartTime_s = 50.0
#clipEndTime_s = 80.0
#clipStartFrame = int(clipStartTime_s*fps)
#clipEndFrame = int(clipEndTime_s*fps)

clipRunTime_s = 60
clipStartTime_s = 20
clipEndTime_s = clipStartTime_s + clipRunTime_s
clipStartFrame = clipStartTime_s*fps
if clipRunTime_s == 0:
    clipRunFrames = int(fCount - clipStartFrame)
else:
    clipRunFrames = int((clipEndTime_s- clipStartTime_s)*fps)
    print(f" Clip run frames: {clipRunFrames}")

#videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame)) # Initial start point, goes to 1st key frame
videoOpbject.set(cv2.CAP_PROP_POS_MSEC, 25*1000) # Initial start point

#frame_timestamp_ms = 0 # Start at the clip start time in milliseconds

clipStartFrame = (fps *40) # Start frame for the clip

frame_timestamp_ms = clipStartFrame * frameTime_ms # Timestamp for the first frame in the clip

videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame)) # Initial start point

remainingFrames = int(fCount - clipStartFrame)

for i in range(int(remainingFrames)): # Go through each frame
#for num in range(clipRunFrames): # Go through each frame
    #frame_timestamp_ms += int(num * frameTime_ms)   increment through the frames
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms) #increments frame timestamp by the time of each frame
    #print(f"Frame timestamp: {frame_timestamp_ms} ms")
    sucess, frame = videoOpbject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    # Check to see if we actualy loaded a frame 
    if not sucess:
        print(f"Frame read failure")
        exit()
    #getDateTime(frame) # Get the date time from the frame

    #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms)) # Detect the pose landmarks in each frame
    #annotated_image = draw_landmarks_on_image(image.numpy_view(), pose_landmarker_result)

    #draws on frames
    if len(pose_landmarker_result.pose_world_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        drawLandmark_circle(frame, landmarks[29]) # Draw circle on the left heel
        drawLandmark_line(frame, landmarks[29],landmarks[23]) # Draws line from left foot to left hip
        drawLandmark_line(frame, landmarks[30], landmarks[24]) # Draws line from right foot to right hip
        drawLandmark_circle(frame, landmarks[30]) # Draw circle on the right heel
    else:
        print(f"No pose detected at frame {i}, time {frame_timestamp_ms} ms")
        
    #Get the location of feet and hips landmarks 
    rthip_x = int(landmarks[24].x * width)
    lfthip_x = int(landmarks[23].x * width)
    rtft_y = int(landmarks[32].y * height)
        # Crop the frame with padding
    new_frame = crop_with_padding(frame, lfthip_x, rthip_x, rtft_y) 
        #print(f"L-H: {pose_landmarker_result.pose_world_landmarks[0][29].y}"
             # f"R-H: {pose_landmarker_result.pose_world_landmarks[0][30].y}") 
    if new_frame.size > 0:
            cv2.imshow("Input", new_frame)
    else:
            print("Invalid crop — skipping frame")

    #print(pose_landmarker_result)
    #Show the frame 

   

    #print(f"Right foot x: {rtft_x}, y: {rtft_y}")

    # Calculate center between left and right hips center_x = int((lfthip_x + rthip_x) / 2)
    
    
    #print(f"The width is: {rthp_x +100 - (lfthp_x - 100)}")

    #display the frame
    
    
    #frame = cv2.resize(new_frame, displayRez) # Resize the frame for display
    #qnew_frame = frame[w-int(landmarks[31].x)*w:h, int(h-landmarks[31].y)*h:h]
    #cv2.imshow("Input", frame)

    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()

  
