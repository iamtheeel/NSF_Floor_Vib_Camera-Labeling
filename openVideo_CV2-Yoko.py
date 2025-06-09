####
# Yoko Lu
# SFSU STARS Summer 2025
# Dr J Lab
###
# Opens a video file in OpenCV and shows frame processing timing
####

# Built-ins
import time

# Third-party
import cv2  # pip install opencv-python
import pytesseract  # pip install pytesseract

# === SETTINGS ===
dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}{file}"  # Full path to the video file

# === Add our new funtion here === "
#Add variable and does not return, no parameters#

#def print_frame_time(name):
    #return "Message is: This is the print_frame_time function"
    #print(f"Message is: This is the print_frame_time function {name}")

def compare_video_fps(video1_fps, video2_fps):
    print(f"Video 1 FPS: {video1_fps} Hz")
    print(f"Video 2 FPS: {video2_fps} Hz")


# === OPEN VIDEO ===
videoObject = cv2.VideoCapture(filename)

if not videoObject.isOpened():
    print(f"❌ Error opening video file: {filename}")
    #exit()



print(f"{dir=}, {filename=}")


# Call the new function here
#print_frame_time("Object")

compare_video_fps("First video fps,", "Second video fps")
video2_fps = 5 # Example FPS for the second video


#exit()

# === VIDEO METADATA ===
fps = videoObject.get(cv2.CAP_PROP_FPS)
fCount = int(videoObject.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = fCount if fCount > 0 else 0  # Ensure frame count is valid

print(f"✅ Loaded: {filename}")
print(f"🎞️ FPS: {fps:.2f} Hz")
print(f"📊 Total Frames: {frame_count}, Type: {type(frame_count)}")
print(f"🖼️ Width: {w}, Height: {h}")

# === TIMING ===
idealFrameDelay_ms = 1000 / fps if fps > 0 else 33.33  # ms per frame
print(f"⏱️ Target Frame Time: {idealFrameDelay_ms:.2f} ms")

# === DISPLAY SETTINGS ===
dispFact = 2
displayRes = (int(w / dispFact), int(h / dispFact))  # Resize for display

# === FRAME LOOP ===
for i in range(frame_count):
    startTime = time.time()  # Start clock

    success, frame = videoObject.read()
    if not success:
        print(f"❌ Failed to read frame {i}")
        break


      # === GRAYSCALE + BLUR ===
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayFrame, (5, 5), 1.4)

    # === CANNY ===
    edges = cv2.Canny(blurred, 10, 10)

    # === LAPLACIAN ===
    lap = cv2.Laplacian(grayFrame, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)

    # === RESIZE FOR DISPLAY ===
    resizedCanny = cv2.resize(edges, displayRes)
    resizedLap = cv2.resize(lap, displayRes)

    # === COMBINE FOR VISUAL COMPARISON ===
    canny_bgr = cv2.cvtColor(resizedCanny, cv2.COLOR_GRAY2BGR)
    lap_bgr = cv2.cvtColor(resizedLap, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([canny_bgr, lap_bgr])

    #cv2.imshow('Canny | Laplacian', combined)

    endTime = time.time()  # End clock
    processingTime_ms = 1000 * (endTime - startTime)  # ms
    actualWaitTime_ms = max(1, idealFrameDelay_ms - processingTime_ms)

    # ✅ Corrected print: show real delay used
    print(f"proc: {processingTime_ms:.2f} ms, delay: {actualWaitTime_ms:.2f} ms")

    key = cv2.waitKey(int(actualWaitTime_ms))
    if key & 0xFF == ord('q'):
        print("🔴 Quit by user.")
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Resize for date/time extraction
    # Resize for date/time extraction

    dateTime_img = frame[0:40, 0:385]   #Get just the time frame = array (...)
    dateTime_img_bw =cv2.cvtColor(dateTime_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    dateTime_img = 255 - dateTime_img_bw #Invert the image
    #print(f"dateTime_img: type: {type(dateTime_img)}, len: {dateTime_img.shape}")
    #print(dateTime_img[0.30, 25:40])  # Print the shape of the date/time image
    dateTime_output = pytesseract.image_to_string(dateTime_img_bw, output_type=pytesseract.Output.DICT)  # Extract text from the date/time image
    print(f"outFrame: type: {type(dateTime_img)}, len: {dateTime_img.shape}")
    dateTime_output = pytesseract.image_to_data(dateTime_img, output_type=pytesseract.Output.DICT)
    object = 5 # Index of the date/time object in the output dictionary (5: time, 0: the whole timeframe)
    pt1 = (dateTime_output["left"][object], dateTime_output["top"][object])
    pt2 = (dateTime_output["width"][object] + dateTime_output["left"][object], 
           dateTime_output["top"][object] + dateTime_output["height"][object])
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(dateTime_img, pt1, pt2, color, thickness)  # Draw rectangle around date/time area

    dateTime_str = dateTime_output['text']
    dateTime_conf = dateTime_output['conf']  # Extract confidence from the date/time image
    print(f"Date: {dateTime_str[4]}: {dateTime_conf[4]}",
          f"Time: {dateTime_str[5]}: {dateTime_conf[5]}")


    cv2.imshow('Video Frame', dateTime_img)


# === CLEANUP ===
videoObject.release()
cv2.destroyAllWindows()
 