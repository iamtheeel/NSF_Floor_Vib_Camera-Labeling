####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Opens a video file in OpenCV
####

import cv2 # pip install opencv-python

# The File
dir = 'StudentData/25_06_03/Subject_1'
file = '25_06_03_s1_1.asf'
fileName = f"{dir}/{file}"


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

# Display
dispFact = 2
displayRez = (int(w/dispFact), int(h/dispFact))
#exit()

for i in range(int(fCount)):
    #print(f"Load frame {i}")
    sucess, frame = videoOpbject.read() #.read() returns a boolean value and the frame itself. 
                                    # sucess (t/f): Did this frame read sucessfully?
                                    # frame:  The video image
    if not sucess:
        print(f"Frame read failure")
        exit()

    lwResFrame = cv2.resize(frame, displayRez)
    cv2.imshow("Frame", lwResFrame)
    #key = cv2.waitKey(0)
    key = cv2.waitKey(int(1000/fps))
    print(key)
    if key == ord('q') & 0xFF: exit()
