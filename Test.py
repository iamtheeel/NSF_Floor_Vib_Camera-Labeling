####
#   Yoko Lu
#   SFSU STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking + Timestamp OCR
####

# === Imports ===
import time
import cv2
import pytesseract  # pip install pytesseract
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === SETTINGS ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}{video_file}"

# === Load Video ===
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = videoObject.get(cv2.CAP_PROP_FPS)
frameTime_ms = 1000 / fps
fCount = int(videoObject.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === MediaPipe Setup ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)

# === Time Clip Settings ===
clipStart_time_s = 10
clipEnd_time_s = 20
start_frame = int(clipStart_time_s * fps)
end_frame = int(clipEnd_time_s * fps)
videoObject.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# === FRAME LOOP for 10s to 20s ===
for i in range(end_frame - start_frame):
    success, frame = videoObject.read()
    if not success:
        print(f"❌ Failed to read frame {i}")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # === MediaPipe Pose Detection ===
    current_frame = start_frame + i
    frame_timestamp_ms = int(current_frame * frameTime_ms)
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # === OCR Timestamp Extraction ===
    timestamp_roi = frame[0:40, 0:400]  # Adjust crop if needed
    gray = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray  # Invert for better OCR
    ocr_result = pytesseract.image_to_string(gray, config='--psm 7')
    ocr_result = ocr_result.strip().replace('\n', '').replace('\x0c', '')
    time_string = ocr_result if ocr_result else "??:??:??"

    # === Output Pose Info ===
    if result.pose_landmarks:
        landmarks = result.pose_landmarks
        confidences = [lm.presence for lm in landmarks if hasattr(lm, 'presence')]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"🕒 {time_string} | Landmarks: {len(landmarks)}, Avg Conf: {avg_conf:.3f}")
    else:
        print(f"🕒 {time_string} | Landmarks: 0, Avg Conf: 0.000")

    # === Show Video Frame ===
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Pose Tracking", frame_bgr)
    cv2.imshow("Timestamp ROI", gray)  # Optional debug window for timestamp area

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("🔴 Quit by user.")
        break

# === Cleanup ===
videoObject.release()
cv2.destroyAllWindows()
