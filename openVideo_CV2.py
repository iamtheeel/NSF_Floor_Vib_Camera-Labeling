####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Opens a video file in OpenCV
####

#Built ins
import time

#Third party
import cv2 # pip install opencv-python
import pytesseract

#in house

# The File
dir = 'StudentData/25_06_03/Subject_1'
file = '25_06_03_s1_1.asf'
fileName = f"{dir}/{file}"

## Add our new function here declare
def foo(bar):
    print("bar")
#Do not call your new function here

# Make a video object
videoOpbject = cv2.VideoCapture(fileName)

if not videoOpbject.isOpened():
    print("Error: Could not open video.")
    exit()

#Call new function here
foo("Object")
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

    outFrame = cv2.resize(frame, displayRez)

    # canny edge detection?

    #  normalize
    #outFrame = cv2.normalize(outFrame, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=-1, mask=None)
    #outFrame = cv2.Laplacian(outFrame, ddepth=cv2.CV_64F)
    dateTime_img = outFrame[0:23, 0:192]
    dateTime_img = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)
    dateTime_img = cv2.bitwise_not(dateTime_img)
    _, dateTime_img = cv2.threshold(dateTime_img, 150, 255, cv2.THRESH_BINARY)

    dateTime_str = pytesseract.image_to_string(dateTime_img)
    print(f"{dateTime_str}")

    cv2.imshow("Frame", dateTime_img)
    tEnd_s = time.time()
    pTime_ms = 1000*(tEnd_s - tStart_s)
    delayTime_ms = frameDelay_ms - pTime_ms
    if delayTime_ms <= 0: delayTime_ms = 1 # make sure we don't have a negitive delay

    #print(f"fDelay: {frameDelay_ms}, proc Time: {pTime_ms}, delay: {delayTime_ms} ms")
    key = cv2.waitKey(int(delayTime_ms))
    #key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()