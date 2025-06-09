import cv2
import mediapipe as mp
import os

# === SETTINGS ===
video_path = "/Users/yokolu/NSF_Floor_Vib_Camera-Labeling/s2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf"
output_path = "/Users/yokolu/NSF_Floor_Vib_Camera-Labeling/output_pose_tracking.mp4"

# === SAFETY CHECK ===
if not os.path.exists(video_path):
    print(f"❌ File not found: {video_path}")
    exit()

# === INIT MEDIAPIPE POSE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

# === VIDEO WRITER ===
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🚀 Processing video... Press 'q' to exit early.")

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done. Output saved to: {output_path}")
