import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pytesseract
import math

# === CONFIGURATION ===
video_file = "/Volumes/MY PASSPORT/Stars_day1Data/s2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf"
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task"  # ⚡ Lite model for speed

# Calibration: adjust based on known object in video
cm_per_pixel = 1 / 2.7  # ← 1 cm = 2.7 pixels

# === VIDEO SETUP ===
video = cv2.VideoCapture(video_file)
if not video.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
frameTime_ms = 1000 / fps
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === MEDIAPIPE SETUP ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

# === OCR FUNCTION ===
def getDateTime(frame):
    crop = frame[0:45, 0:384]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    data = pytesseract.image_to_data(inv, output_type=pytesseract.Output.DICT)
    try:
        date_str = data['text'][4]
        time_str = data['text'][5]
        conf = int(data['conf'][4])
        return f"{time_str}", conf
    except:
        return "OCR Error", 0

# === MAIN LOOP ===
frame_idx = 0
last_timestamp = "Unknown"
last_conf = 0

log_file = open("heel_stride_log.txt", "w")

with PoseLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = video.read()
        if not success:
            break

        # 🔁 Skip every other frame for speed
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(frame_idx * frameTime_ms)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Get heel coordinates
            l_heel = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value]
            r_heel = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value]

            lx, ly = int(l_heel.x * w), int(l_heel.y * h)
            rx, ry = int(r_heel.x * w), int(r_heel.y * h)

            heel_px = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
            heel_cm = heel_px * cm_per_pixel
            heel_m = heel_cm / 100

            # 🕒 OCR every 10 frames
            if frame_idx % 10 == 0:
                last_timestamp, last_conf = getDateTime(frame)

            conf_str = f"{last_conf}%" if last_conf > 0 else "N/A"

            # 📄 Save to log
            log_file.write(f"🕒 Time: {last_timestamp} | conf: {conf_str}\n")
            log_file.write(f"📏 Frame {frame_idx}: Heel distance = {heel_m:.4f} m ({heel_cm:.1f} cm)\n")

        # 🖼️ GUI disabled for speed
        # cv2.imshow("Pose Heel Measurement", frame)  # <- Skipped

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

video.release()
cv2.destroyAllWindows()
log_file.close()
print("✅ Done. Results saved to heel_stride_log.txt")
