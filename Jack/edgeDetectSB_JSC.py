####
#   Jack Capito
#   STARS Summer 2025
#   Dr J Lab
###
# Edgetracing experiment
####

import cv2 # pip install opencv-python
import time
import numpy as np

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_06_03\Subject_2'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}/{file}" # Path to the video file
print(filename)

videoObject = cv2.VideoCapture(filename) # Open the video file
if not videoObject.isOpened():
    print("Error: Could not open video file.")
    exit()
print(f"Loaded: {filename}")

def perameters(text, videoPeram):
    output = videoObject.get(videoPeram)
    print(f"Loaded: {output}{text}")
    return output

fps = perameters("fps", cv2.CAP_PROP_FPS)
vidHeight = perameters("px height", cv2.CAP_PROP_FRAME_HEIGHT)
vidWidth = perameters("px width", cv2.CAP_PROP_FRAME_WIDTH)
frameCount = perameters(" frames in frameCount", cv2.CAP_PROP_FRAME_COUNT)
dispFact=2
displayRez = (int(vidWidth/dispFact), int(vidHeight/dispFact)) # Calculate the display resolution by dividing the width and height by a factor


for i in range(int(frameCount)): 
    #print(f"load frame {i}")
    start_time = time.time()  # Start timing

    success, frame = videoObject.read() # .read() returns a boolean value and the frame itself. 
                                        #success (t/f): did the frame read successfull?
                                        #frame: the video image
    #check to see if we actually loaded a frame
    if not success:
        print(f"frame read failure")
        exit()

    
    blackWhite = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    lines = cv2.Canny(blackWhite, 80, 100)
    lwresframe = cv2.resize(lines, displayRez)

    cv2.imshow("Video Frame", lwresframe) # Display the frame in a window named "Video Frame"



    processing_time = (time.time() - start_time) * 1000  # in milliseconds
    delay = max(int(1000/fps) - int(processing_time), 1)  # Ensure at least 1 ms delay

    #print(f"Processing time: {processing_time:.2f} ms, Delay: {delay} ms")  # Optional: see timing info

    # stop the clock
    key = cv2.waitKey(delay) # Wait for a key press for a duration based on the video's FPS
    if key == ord('q') & 0xFF: exit() # If the 'q' key is pressed, exit the loop
