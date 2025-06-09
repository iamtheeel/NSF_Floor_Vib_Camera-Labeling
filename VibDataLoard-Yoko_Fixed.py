####
#   Yoko Lu
#   SFSU STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking with Real-World Clock Overlay
####

import time
import cv2
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
fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT)
w = videoObject.get(cv2.CAP_PROP_FRAME_WIDTH)
h = videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

# === Real-World Clock Start (from video overlay time) ===
video_start_time_s = 4 * 3600 + 0 * 60 + 3  # 04:00:03

# === Time Clip Settings (10s to 20s) ===
clipStart_time_s = 10
clipEnd_time_s = 20
start_frame = int(clipStart_time_s * fps)
end_frame = int(clipEnd_time_s * fps)
videoObject.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# === Frame Loop ===
for i in range(end_frame - start_frame):
    success, frame = videoObject.read()
    if not success:
        print(f"❌ Failed to read frame {i + start_frame}")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # === MediaPipe Pose Estimation ===
    current_frame = start_frame + i
    frame_timestamp_ms = int(current_frame * frameTime_ms)
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # === Real-World Time Calculation ===
    seconds_from_start = current_frame / fps
    total_seconds = int(video_start_time_s + seconds_from_start)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    time_string = f"{hours:02}:{minutes:02}:{seconds:02}"

    # === Pose Output ===
    if result.pose_landmarks:
        landmarks = result.pose_landmarks
        confidences = [lm.presence for lm in landmarks if hasattr(lm, 'presence')]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"{time_string} | Landmarks: {len(landmarks)}, Avg Conf: {avg_conf:.3f}")
    else:
        print(f"{time_string} | Landmarks: 0, Avg Conf: 0.000")

    # === Display Frame ===
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Pose Tracking", frame_bgr)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("🔴 Quit by user.")
        break

# === Cleanup ===
videoObject.release()
cv2.destroyAllWindows()
