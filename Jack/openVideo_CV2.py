####
#Jack Capito
#STARS Summer 2025
#Dr J Lab
###
#opens a video file in CV2
####

import cv2 # pip install opencv-python
import time
import pytesseract # pip install pytesseract

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_06_03\Subject_2'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}/{file}" # Path to the video file
print(filename)

videoObject = cv2.VideoCapture(filename) # Open the video file
if not videoObject.isOpened():
    print("Error: Could not open video file.")
    exit()
print(f"Loaded: {filename}")

def nothingburger(text, videoPeram):
    output = videoObject.get(videoPeram)
    print(f"Loaded: {output}{text}")
    return output

fps = nothingburger("fps", cv2.CAP_PROP_FPS)
vidHeight = nothingburger("px height", cv2.CAP_PROP_FRAME_HEIGHT)
vidWidth = nothingburger("px width", cv2.CAP_PROP_FRAME_WIDTH)
frameCount = nothingburger(" frames in frameCount", cv2.CAP_PROP_FRAME_COUNT)
zoom = nothingburger(" zoom", cv2.CAP_PROP_ZOOM)
pan = nothingburger(" pan", cv2.CAP_PROP_PAN)
tilt = nothingburger(" tilt", cv2.CAP_PROP_TILT)
dispFact=2
displayRez = (int(vidWidth/dispFact), int(vidHeight/dispFact)) # Calculate the display resolution by dividing the width and height by a factor


for i in range(int(frameCount)): 
    print(f"load frame {i}")
    start_time = time.time()  # Start timing

    success, frame = videoObject.read() # .read() returns a boolean value and the frame itself. 
                                        #success (t/f): did the frame read successfull?
                                        #frame: the video image
    #check to see if we actually loaded a frame
    if not success:
        print(f"frame read failure")
        exit()

   
    lwresframe = cv2.resize(frame, displayRez)
    #blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    blackWhite = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white2Black = cv2.bitwise_not(blackWhite)

    dateTime_img = white2Black[0:45, 0:384]
    #print(f"frame: type: {type(dateTime_img)}, len {dateTime_img.shape}")
    dateTimeOutput = pytesseract.image_to_data(dateTime_img, output_type=pytesseract.Output.DICT)
    dateTime_str = dateTimeOutput['text']
    dateTime_conf = dateTimeOutput['conf']
    print(f"Date: {dateTime_str[4]}, Conf:{dateTime_conf[4]}",
          f"Time: {dateTime_str[5]}, Conf:{dateTime_conf[5]}") 
    

    cv2.imshow("Video Frame", dateTime_img) # Display the frame in a window named "Video Frame"



    processing_time = (time.time() - start_time) * 1000  # in milliseconds
    delay = max(int(1000/fps) - int(processing_time), 1)  # Ensure at least 1 ms delay

    print(f"Processing time: {processing_time:.2f} ms, Delay: {delay} ms")  # Optional: see timing info

    # stop the clock
    key = cv2.waitKey(delay) # Wait for a key press for a duration based on the video's FPS
    if key == ord('q') & 0xFF: exit() # If the 'q' key is pressed, exit the loop
