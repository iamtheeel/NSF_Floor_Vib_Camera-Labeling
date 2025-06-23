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
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_lite.task" # 5.5 MiB
#model_path = r"C:\Users\smitt\STARS\pose_landmarker_full.task" # 9.0 MiB
model_path = r"C:\Users\smitt\STARS\pose_landmarker_heavy.task" # 29.2 MiB


#Video File
dir = r"E:\STARS\day1_data"
file = r"25_06_03_s1_1.asf"
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
#height, width, _ = videoOpbject.read()[1].shape doing this reads the first frame, which we don't want to do yet
width = int(videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH)) # Width of the video frame
height = int(videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of the video frame
#width = 256
#height = 256 

frameTime_ms = 1000/fps #How long of a time does each frame cover
# Fit to the display
dispFact = 2
displayRez = (int(width/dispFact), int(height/dispFact))

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
                                running_mode=VisionRunningMode.VIDEO,
                               )

landmarkerVideo = PoseLandmarker.create_from_options(options)
#exit()

#functions

def drawLandmark_circle(frame, center_width, center_height):
    radius = 15
    thickness = 5
    color = [255, 0, 0] #Circle will be red
    center = int(center_width), int(center_height)
    #center = int(landmark.x*width), int(landmark.y*height) #place center of the circle at the landmark position
    #print(f"x: {int(landmark.x*w)}, y: {int(landmark.y*h)}")q
    cv2.circle(frame, center, radius, color, thickness) # Draw a circle at the landmark position
    
def drawLandmark_square(frame, minWidth, maxWidth, minHeight, maxHeight):
    color = [255, 0, 0] # Line will be red
    #pt1_ft = (int(feet.x*width),int(feet.y*height)) #First point is on the feet
    #pt2_hips = (int(hips.x*width), int(hips.y*height)) #second point is on the hips
    xyPt = int(minWidth),int(minHeight) #upper left pt
    XyPt = int(maxWidth), int(minHeight) #upper right pt
    XYPt = int(maxWidth), int(maxHeight) #lower right pt
    xYPt = int(minWidth), int(maxHeight) #lower left pt
    thickness = 5
    #Connects points to draw a square
    cv2.line(frame, xyPt, XyPt, color, thickness) 
    cv2.line(frame, XyPt, XYPt, color, thickness) #
    cv2.line(frame, XYPt, xYPt, color, thickness)
    cv2.line(frame, xYPt, xyPt, color, thickness)

def isPersonInFrame(frame, frameIndex):
    #videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, frame_Index) # Set the video object to the frame we want to check
    #ret, frame = videoOpbject.read() # Read the frame at the specified index
    #if not ret: # If the frame was not read successfully, return None
    #    print("Error: Could not read frame.")
    #    return None
    
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convert frame from BGR to RGB
    frame_timestamp_ms = int(frameIndex * frameTime_ms) 
    if frame_timestamp_ms < 0 or frame_timestamp_ms > 1e10: # Check if the timestamp is valid
        #print(f"Invalid timestamp: {frame_timestamp_ms}")
        return None #Exit function if the timestamp is invalid

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_RGB) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarkerVideo.detect_for_video(mp_image, frame_timestamp_ms) #Detect the pose landmarks in the frame
    #If there are no pose landmarkers
    if len(pose_landmarker_result.pose_landmarks) > 0: 
        #print(f"ISPERSON FUNCTION: for frame index: {frame_Index} with frame: {videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)}") # If there is a pose landmarker, return True
        #print(f"There is a person!")
        print(f"RAW_FILE visibility for frame {frameIndex} is {pose_landmarker_result.pose_landmarks[0][30].visibility}")
        return True, pose_landmarker_result
    else:
        return False, None
    #else:
        #print(f"No person detected at {frame_timestamp_ms} ms")
        # No person detected, handle accordingly
    #return False # If there is no pose landmarker, return False
     #pose_landmarker_result

def crop_with_padding(frame, landmarks):
    #Checks if there are landmarkers 
    
    #landmarks = pose_landmarker_result.pose_landmarks[0]

    frame_height, frame_width = frame.shape[:2] 

    min_width = max_width = landmarks[0].x #Initiates width variables to landmark 0
    min_height = max_height = landmarks[0].y #Initiates height variables to landmark 0
    min_width_index = max_width_index = min_height_index = max_height_index = 0

    #Iterates through all landmarks to find max and min: x value = width, y value = height
    for i in range(len(landmarks)):
        x = landmarks[i].x
        y = landmarks[i].y

        if x < min_width:
            min_width = x
            min_width_index = i
        if x > max_width:
            max_width = x
            max_width_index = i
        if y < min_height:
            min_height = y
            min_height_index = i
        if y > max_height:
            max_height = y
            max_height_index = i


    #Normalize values to frame
    min_width = min_width*frame_width
    max_width = max_width*frame_width
    min_height=min_height*frame_height
    max_height=max_height*frame_height
    #print(f"CROPWITHPADDING function: The MINIMUM width is {min_width} at index {min_width_index}")
    #print(f"CROPWITHPADDING function: The MAXIMUM width is {max_width} at index {max_width_index}")
    #print(f"CROPWITHPADDING function: The MINIMUM height is {min_height} at index {min_height_index}")
    #print(f"CROPWITHPADDING function: The MAXIMUM height is {max_height} at index {max_height_index}")

    tot_width = max_width - min_width   #total width of cropped frame
    tot_height = max_height - min_height #total height of cropped frame
    
    #print(f"CROPWITHPADDING function: The TOTAL width is {tot_width}")
    #print(f"CROPWITHPADDING function: The TOTAL height is {tot_height}")
    
    #expand_width = (1*tot_width*frame_width)//2
    #expand_height = (1*tot_height*frame_height)//2

    #Padded_minwidth = min_width*frame_width - expand_width
    #Padded_maxwidth = max_width*frame_width + expand_width

    #Padded_minheight = min_height*frame_height - expand_height
    #Padded_maxheight = max_height*frame_height + expand_height

    #total_widthPadded = Padded_maxwidth - Padded_minwidth
    #total_heightPadded = Padded_maxheight - Padded_minheight
    
    Ratio = width/height # Ratio of height/width of full sized frame

    current_ratio = tot_width / tot_height #Ratio of height/width of cropped frame
    
    #Finds the center WRT full frame by adding half of width/height of 
    #cropped screen to min height/width in full frame dimensions
    center_width = min_width + tot_width / 2 
    center_height = min_height + tot_height / 2

    #Change height/width ratio of cropped frame to match that of full frame
    if current_ratio < Ratio:
    # Too narrow: increase width (or crop height)
        adjust_width = (tot_height * Ratio) / 2
        min_width = center_width - adjust_width
        max_width = center_width + adjust_width
        #print(f"Width adjusted. Min width {min_width}. Max width {max_width}")
    else:
    # Too wide: crop width (or increase height)
        adjust_height = (tot_width / Ratio) / 2
        min_height = center_height - adjust_height
        max_height = center_height + adjust_height
        #print(f"Height adjusted. Min height {min_height}. Max height {max_height}")
    
    #adjusts total width/height according to new dimensions
    tot_width = max_width - min_width
    tot_height = max_height - min_height
    
    #adjusts center according to new dimensions
    center_width = min_width + tot_width / 2
    center_height = min_height + tot_height / 2
    
    scale_factor = 1.5

    #scales total width/height
    new_width = tot_width * scale_factor
    new_height = tot_height * scale_factor

    #Calculates new min/max height/width WRT to full frame by adding 
    #center (in full frame coords) to width/height (in cropped frame coords)
    min_width = center_width - new_width / 2
    max_width = center_width + new_width / 2
    min_height = center_height - new_height / 2
    max_height = center_height + new_height / 2
    #if tot_height > adjust_height: 
      #  min_width = (center_width - adjust_width)
      #  max_width = (center_width + adjust_width)
    #    print(f"If statement initialised. Min width {min_width}. Max width {max_width}")
    #else:
     #   min_height = center_height - adjust_height
     #   max_height = center_height + adjust_height



    #Padded_maxwidth = 1.1*max_width*frame_width
    #Padded_minheight = .8*min_height*frame_height
    #Padded_maxheight = 1.1*max_height*frame_height

    #min_width_loc = frame_width*np.min(landmarks.x)
    #print(f"CROPWITHPADDING Frame width: {frame_width}.")
 
    #Horizontal landmarks WRT frame width
    #rightHip = int(landmarks[24].x * width)
    #lefttHip = int(landmarks[23].x * width)
    #print(f"CROPWITHPADDING Right hip: {rightHip}. Left hip: {lefttHip}.")
    #crop_width = frame_width #/ dispFact
    #hip_x = (rightHip + lefttHip) // 2 # Calculate horizontal center between left and right hips 
    #print(f"CROPWITHPADDING center distance:{hip_x}.")
    #Vertical landmarks
    #rightShoulder = int(landmarks[12].y * height) #Gets coord of right should WRT to width of screen
    #lefttShoulder = int(landmarks[11].y * height) #Gets coord of left should WRT to width of screen
    
    # Set horizontal crop bounds equidistant from the center of hips
    #crop_width = frame_width/dispFact

    #new_minwidth = hip_x - FIXED_CROP_WIDTH // 2
    #new_maxwidth = hip_x + FIXED_CROP_WIDTH // 2

    #print(f"CROPWITHPADDING_FIRST function. Cropwidth: {crop_width}")
    #print(f"CROPWITHPADDING function. Minwidth: {new_minwidth}. Padded maxwidth: {new_maxwidth}.")

    # Calculate how much we are out of bounds
    #If x1 / x2 are out of bounds (maximum), we need to pad the frame
     #pad_left   = max(0, -new_minwidth) #x1 is negative if it extends beyond the left edge of the frame
     #pad_right  = max(0, new_maxwidth - frame_width) #x2 is positive if it extends beyond the right edge of the frame
        
    #Apply padding if needed
    #frame_padded = cv2.copyMakeBorder(frame,0,0,pad_left,pad_right,borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))  # Black padding
    #Add padding to new frame bounds if needed
     #padded_minwidth= int(new_minwidth + pad_left)
     #padded_maxwidth = int(new_maxwidth + pad_right)
    
    #print(f"CROPWITHPADDING_FIRST function. Padded minwidth: {padded_minwidth}. Paddded maxwidth: {padded_maxwidth}.")
    #Saves visibility data for right heel and right shoulder
    #head_visibility = landmarks[12].visibility
    #right_hip_visibility = landmarks[32].visibility

    #if right_hip_visibility > head_visibility:
       # print(f"Bottom half of person")
    #else:
        #print(f"Top half of person")
    #y1_padded = y1 + pad_top
    #y2_padded = y2 + pad_top
    # Crop the padded frame
    #new_frame = frame_padded[:, x1_padded:x2_padded]
    #print(f"Crop bounds: x1: {x1_padded}, x2: {x2_padded}")

    #Ensures that crop is within bounds (0 to full frame size)
    min_width = max(0, round(min_width))
    max_width = min(frame_width, round(max_width))
    min_height = max(0, round(min_height))
    max_height = min(frame_height, round(max_height))

    # Make sure the result isn't an empty crop
    if max_width <= min_width or max_height <= min_height:
    # Return the full frame as fallback
        return 0, frame_width, 0, frame_height
    
    return round(min_width), round(max_width), round(min_height), round(max_height)

# Main code
start_frame = 1500 # Start frame for the clip
end_frame = int(fCount) 
print("Initial frame position:", videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)) #Ensures initial frame is 0

# Read frames until we reach the frame prior to start frame
videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

#initializes height/width to full size
max_height = height
min_height = 0
max_width = width
min_width = 0

#Opens 
with open(r'E:\STARS\output.txt', 'a', encoding="utf-8") as f:
    #f.write("Test data for Cut Frame Data for 25_06_03_s1_1.asf frames 1642-2100 left heel visibility:\n")
    #f.write("\n")
    #Read through the specified frame count
    for frame_Index in range(start_frame, end_frame): 
        success, raw_frame = videoOpbject.read() # Returns a boolean and the next frame
        if not success: # If the frame was not read successfully, break the loop
            print("Failed to read frame")
            exit()
        newDim_Frame = raw_frame[min_height:max_height,min_width:max_width] #Taking a full sized frame and 
    #Shrinking it down to dimensions
    #Changes dimensions before finding landmarks
    
        #center_w = (min_width + (max_width - min_width)/2)
        #center_h = (min_height + (max_height - min_height)/2)
    #drawLandmark_circle(raw_frame, center_w, center_h)
        drawLandmark_square(raw_frame, min_width, max_width, min_height, max_height) #Returns a box around the person

        resizedFrame = cv2.resize(raw_frame, displayRez) # Resize the frame for displayd
        cv2.imshow("Frame", resizedFrame) #displays frame
        key1 = cv2.waitKey(0) # Wait for a key press

        #resizedCropFrame = cv2.resize(newDim_Frame, displayRez) # Resize the frame for displayd
        #cv2.imshow("Frame", resizedCropFrame) #displays frame
        #key2 = cv2.waitKey(1) # Wait for a key press
        key2 = 0
        if key1 == ord('q') or key2 == ord('q') & 0xFF: exit()
        good = False

        if raw_frame is not None: #Failsafe "if newDim_Frame is not None:"
        #good, result = isPersonInFrame(newDim_Frame, frame_Index) #newDim_Frame Checks if there is a person in the frame. Returns frame and landmarkers.
        #rescale and reshift
            good, result = isPersonInFrame(newDim_Frame, frame_Index) #newDim_Frame Checks if there is a person
            if good and result is not None:
                landmarks = result.pose_landmarks[0]
                #f.write(f"Frame index: {videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)}")
                #f.write("\n")
                #f.write("Visibility: {landmarks[30].visibility}")
                #if.write("\n")
                for i in range(len(landmarks)):
                #landmarks[i].x = landmarks[i].x*(width/(max_width-min_width))
                #landmarks[i].x =  landmarks[i].x*width - (center_w- width/2)
                #landmarks[i].x = landmarks[i].x/width 

                #landmarks[i].y = landmarks[i].y*(height/(max_height-min_height))
                #landmarks[i].y =  landmarks[i].y*height - (center_h- height/2)
                #landmarks[i].y = landmarks[i].y/height 
                    landmarks[i].x = (landmarks[i].x * (max_width - min_width) + min_width) / width
                    landmarks[i].y = (landmarks[i].y * (max_height - min_height) + min_height) / height
        

    
                min_width, max_width, min_height, max_height = crop_with_padding(raw_frame, landmarks) #, landmarks
            #print(f"BACK IN MAIN for frame: {videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)} Minwidth: {min_width}. Maxwidth: {max_width} ")
            #new_Frame = crop_with_padding(raw_frame, landmarks) #Returns cropped frame
            #resizedFrame = cv2.resize(new_Frame, displayRez) # Resize the frame for display
            else:
                #f.write(f"BACK IN MAIN BUT NOT GREAT for frame: {videoOpbject.get(cv2.CAP_PROP_POS_FRAMES)}")
                #f.write("\n")
                print(f"No person detected for frame index: {frame_Index}")
                min_height = 0
                max_height = height
                min_width = 0
                max_width = width
   


#if isPersonInFrame(frame_Index):
   # print("Person detected in the first frame.")
#else:
    #print("No person detected in the first frame.")



"""
startTime = 30

clipStartFrame = (fps * startTime) # Start frame for the clip

#frame_timestamp_ms = clipStartFrame * frameTime_ms # Timestamp for the first frame in the clip

#videoOpbject.set(cv2.CAP_PROP_POS_FRAMES, int(clipStartFrame)) # Initial start point

remainingFrames = int(fCount - clipStartFrame)

for i in range(int(remainingFrames)): # Go through each frame
    success, frame = videoOpbject.read() # Read the next frame returns a boolean and the frame
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms) # Update the timestamp for the current frame
    if not success: # If the frame was not read successfully, break the loop
        print("Failed to read frame")
        break
    
    #frame = frame[0:400, 1000:1444, :] # Crop the frame to the area of interest
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create a MediaPipe image from the frame
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms)) # Detect the pose landmarks in each frame
    
    if len(pose_landmarker_result.pose_world_landmarks) > 0:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        drawLandmark_circle(frame, landmarks[29]) # Draw circle on the left heel
        drawLandmark_line(frame, landmarks[29],landmarks[23]) # Draws line from left foot to left hip
        drawLandmark_line(frame, landmarks[30], landmarks[24]) # Draws line from right foot to right hip
        drawLandmark_circle(frame, landmarks[30]) # Draw circle on the right heel
       
    if pose_landmarker_result.pose_landmarks:
        for i, landmark in enumerate(pose_landmarker_result.pose_landmarks[0]):
            if landmark.visibility > 0.5:  # Or even 0.3 for partials
                print(f"Landmark {i} detected at ({landmark.x:.2f}, {landmark.y:.2f})")
       
    else:
        print(f"No pose detected at frame {i}, time {frame_timestamp_ms} ms")
    resizedFrame = cv2.resize(frame, displayRez) # Resize the frame for display
    cv2.imshow("Frame", resizedFrame)
    
    key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()
 """