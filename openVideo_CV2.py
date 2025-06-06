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
import pytesseract #pip install pytesseract

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



    #  Get the time
    dateTime_img = frame[0:46, 0:384, :] # Get just the time
    dateTime_img_bw = cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY) # Convert to grey scale
    dateTime_img_bw = 255 - dateTime_img_bw #Invert the image
    #print(f"dateTime_img type: {type(dateTime_img)}, shape: {dateTime_img.shape}")
    #print(dateTime_img[0:30, 25:40])
    dateTime_outPut = pytesseract.image_to_data(dateTime_img_bw, output_type=pytesseract.Output.DICT)
    print(f"outFame: type: {type(dateTime_outPut)}")
    object = 5
    pt1 = [dateTime_outPut['left'][object], dateTime_outPut['top'][object]]
    pt2 = [dateTime_outPut['width'][object] + dateTime_outPut['left'][object], 
           dateTime_outPut['top'][object]+ dateTime_outPut['height'][object]]
    cv2.rectangle(dateTime_img, pt1, pt2, color=(255, 0, 0), thickness=1)
    dateTime_img = cv2.cvtColor(dateTime_img, cv2.COLOR_RGB2BGR) # Convert to grey scale
    print(f": {dateTime_outPut['text'][object]}: {dateTime_outPut['conf'][object]}") 


    #outFrame = cv2.resize(frame, displayRez)
    cv2.imshow("Overlay", dateTime_img)
    cv2.imshow("Recoc", dateTime_img_bw)
    tEnd_s = time.time()
    pTime_ms = 1000*(tEnd_s - tStart_s)
    delayTime_ms = frameDelay_ms - pTime_ms
    if delayTime_ms <= 0: delayTime_ms = 1 # make sure we don't have a negitive delay

    #print(f"fDelay: {frameDelay_ms}, proc Time: {pTime_ms}, delay: {delayTime_ms} ms")
    key = cv2.waitKey(int(delayTime_ms))
    #key = cv2.waitKey(int(1))
    if key == ord('q') & 0xFF: exit()