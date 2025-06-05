import cv2 #import OpenCV lib
import mediapipe as mp #import model
from mediapipe.tasks import python #imports submodules from mediapipe
from mediapipe.tasks.python import vision #imports submodules from mediapipe


Run1 = cv2.VideoCapture("E:\\STARS\\day1_data\\Kara_2_NS_3.40pm.mp4") #Saves vid 1 as object
if Run1.isOpened(): #Checks if the file is opened
    print("Video 1 file opened successfully") #Prints success message if file is opened
else:
    print("Error opening video 1 file") #Prints error if file is not opened

i = 0

while True: #While loop to read the video
    more_frames, frame = Run1.read() #.read() returns a boolean value and the frame itself. more_frames is true if there are more frames to read
    if not more_frames: #if no more frames. more_frames is false and the loop breaks
        break
  



Run2 = cv2.VideoCapture("E:\\STARS\\day1_data\\Kara_3_SN_3.42pm.mp4") #Saves vid 2 as object
if Run2.isOpened(): #Checks if the file is opened
    print("Video 2 file opened successfully") #Prints success message if file is opened
else:
    print("Error opening video 2 file") #Prints error if file is not opened
    