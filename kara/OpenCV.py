####
#   Kara's OpenCV Video Player
#   STARS Summer 2025
#   Dr J Lab
###
# Opens a video file in OpenCV
####

#Built ins
import time

#Third party
import cv2 # pip install opencv-python
import pytesseract # pip install pytesseract
import mediapipe # pip install mediapipe


#in house

# The File
dir = r'E:\STARS\day1_data'
file = r'25_06_03_s1_1.asf'
fileName = f"{dir}/{file}"

# Make a video object
videoOpbject = cv2.VideoCapture(fileName)

if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

fps = videoOpbject.get(cv2.CAP_PROP_FPS)
fCount = videoOpbject.get(cv2.CAP_PROP_FRAME_COUNT)
w = videoOpbject.get(cv2.CAP_PROP_FRAME_WIDTH)
h = videoOpbject.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Loded: {fileName}, FramesPerS: {fps}hz")
print(f"N Frames: {fCount}, type: {type(fCount)}")
print(f"N Width: {w}, height: {h}")

frameDelay_ms = 1000/fps
# Display
dispFact = 2
displayRez = (int(w/dispFact), int(h/dispFact))
#exit()

for i in range(int(fCount)): # Go through each frame
    # We need the processing time to make the correct delay for real time playback
    # start clock
    tStart_s = time.time()
    #print(f"Load frame {i}")
    sucess, frame = videoOpbject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    # Check to see if we actualy loaded a frame 
    if not sucess:
        print(f"Frame read failure")
        exit()

    #Change the resulution to fit
    lwResFrame = cv2.resize(frame, displayRez)
    dateTime_img = frame[0:46,0:400] # Get the date time image from the top left corner
    dateTime_img_bw = 255 - dateTime_img
    dateTime_outPut = pytesseract.image_to_data(dateTime_img, output_type= pytesseract.Output.DICT) # Use pytesseract to read the date time
    cv2.imshow("Date Time",  dateTime_img) # Show the date time image
    print(f"{dateTime_outPut["text"][4]}")
    print(f"outFrame: type: {type(dateTime_outPut)}")
    object = 5
    pt1 = [dateTime_outPut['left'][object], dateTime_outPut['top'][object]]
    pt2 = [dateTime_outPut['left'][object] + dateTime_outPut['left'][object], 
            dateTime_outPut['top'][object] + dateTime_outPut['height'][object]]
    cv2.rectangle(dateTime_img,pt1,pt2,(255,0,0),3)
    cv2.imshow("Frame", lwResFrame)
    tEnd_s = time.time()
    pTime_ms = 1000*(tEnd_s - tStart_s)
    delayTime_ms = frameDelay_ms - pTime_ms
    if delayTime_ms <= 0: delayTime_ms = 1 # make sure we don't have a negitive delay

 
    #print(f"fDelay: {frameDelay_ms}, proc Time: {pTime_ms}, delay: {delayTime_ms} ms")
    key = cv2.waitKey(int(0))
    #key = cv2.waitKey(int(1))
    if key == ord('q'):
        break