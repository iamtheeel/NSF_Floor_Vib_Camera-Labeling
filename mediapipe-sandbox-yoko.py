####
#   Jack Capito
#   STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking with OCR Timestamp (Clean Output)
####

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pytesseract

# === CONFIGURATION ===
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

# === OPEN VIDEO ===
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

nFrames = int(videoObject.get(cv2.CAP_PROP_FRAME_COUNT))  #how many frames?
width  = videoObject.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
fps = videoObject.get(cv2.CAP_PROP_FPS)  # float `height`
fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT)
w = videoObject.get(cv2.CAP_PROP_FRAME_WIDTH)
h = videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = videoObject.get(cv2.CAP_PROP_FPS)

frameTime_ms = 1000 / fps # Convert FPS to milliseconds per frame
startSecs = 125
endSecs = 155

dispFact = 2
displayRez = (int(w/dispFact), int(h/dispFact))

start_frame = int(fps * startSecs)
end_frame = int(fps * endSecs)
num_frames = end_frame - start_frame

if num_frames <= 0:
    print("❌ Invalid clip duration.")
    exit()

videoObject.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# === MediaPipe Setup ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)

# === OCR FUNCTION ===
def getDateTime(frame):
    crop = frame[0:45, 0:384]  # Adjust if needed
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    datetime_output = pytesseract.image_to_data(inv, output_type=pytesseract.Output.DICT)
    try:
        date_str = datetime_output['text'][4]
        time_str = datetime_output['text'][5]
        conf = datetime_output['conf'][4]  # Use only one confidence value (e.g., for date)
        return f"{date_str} {time_str}", conf
    except:
        return "OCR Error", 0

# === MAIN LOOP ===
for i in range(num_frames):
    success, frame = videoObject.read()
    if not success:
        print("❌ Frame read failure")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    frame_timestamp_ms = int((start_frame + i) * frameTime_ms)

    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    timestamp_text, conf = getDateTime(frame)
    print(f"Date: {timestamp_text}, conf: {conf}")

    # Optional: Display frame
    cv2.imshow("Video Frame Only", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("🔴 Quit by user.")
        break

# === CLEANUP ===
videoObject.release()
cv2.destroyAllWindows()
