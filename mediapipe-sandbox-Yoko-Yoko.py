import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    VisionRunningMode
)

# === CONFIGURATION ===
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

stride_history = []

# === OPEN VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_time_ms = 1000 / fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === SETUP MEDIAPIPE TASKS POSE LANDMARKER ===
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)
landmarker = PoseLandmarker.create_from_options(options)

# === PROCESS EACH FRAME ===
frame_index = 0
while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR → RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Pose detection
    timestamp = int(frame_index * frame_time_ms)
    result = landmarker.detect_for_video(mp_image, timestamp)

    # Draw pose + measure stride
    if result.pose_landmarks:
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )

        landmarks = result.pose_landmarks

        # Get left and right heel landmarks
        left_heel = landmarks[29]
        right_heel = landmarks[30]

        # Measure stride length in X (normalized) → convert to pixels
        stride_length_norm = abs(left_heel.x - right_heel.x)
        stride_length_px = stride_length_norm * frame_width

        # Store stride in history
        stride_history.append(stride_length_px)
        if len(stride_history) > 20:
            stride_history.pop(0)

        # Draw stride length text
        cv2.putText(frame, f"Stride: {stride_length_px:.1f}px", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optional: print to terminal
        print(f"[Frame {frame_index}] Stride Length: {stride_length_px:.1f}px")

    # Show the video frame
    cv2.imshow("Pose + Stride Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
landmarker.close()
