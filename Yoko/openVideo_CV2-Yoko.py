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

# === SETTINGS ===
dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}{file}"  # Full path to the video file

# === OPEN VIDEO ===
videoObject = cv2.VideoCapture(filename)

if not videoObject.isOpened():
    print(f"❌ Error opening video file: {filename}")
    exit()

print(f"{dir=}, {filename=}")

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

    resizedFrame = cv2.resize(frame, displayRes)
    cv2.imshow('Video Frame', resizedFrame)

    endTime = time.time()  # End clock
    processingTime_ms = 1000 * (endTime - startTime)  # ms

    actualWaitTime_ms = max(1, idealFrameDelay_ms - processingTime_ms)

    # ✅ Corrected print: show real delay used
    print(f"proc: {processingTime_ms:.2f} ms, delay: {actualWaitTime_ms:.2f} ms")

    key = cv2.waitKey(int(actualWaitTime_ms))
    if key & 0xFF == ord('q'):
        print("🔴 Quit by user.")
        break

# === CLEANUP ===
videoObject.release()
cv2.destroyAllWindows()
 