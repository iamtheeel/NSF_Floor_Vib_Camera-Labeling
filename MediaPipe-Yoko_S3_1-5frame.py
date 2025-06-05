import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

# === SETTINGS ===
dataFile = "/Volumes/MY PASSPORT/Stars_day1Data/Yoko_s3_1.hdf5"
chToPlot = [1, 2, 3, 4, 5]
dataTimeRange_s = [0, 1]  # 1 second only
output_dir = "output_frames"
video_file = "/Volumes/MY PASSPORT/Stars_day1Data/output_trials-S3_1-5.mp4"

# === STEP 1: LOAD DATA ===
def load_data(dataFile, trial=0):
    with h5py.File(dataFile, 'r') as h5file:
        data = h5file['experiment/data'][trial, :, :]
        params = h5file['experiment/general_parameters'][:]
        rate = params[0]['value']
        return data, int(rate)

# === STEP 2: SLICE SENSOR CHANNELS ===
def slice_data(dataBlock, chList, timeRange_sec, rate):
    ch_idx = [ch - 1 for ch in chList]
    start = int(timeRange_sec[0] * rate)
    end = int(timeRange_sec[1] * rate)
    return dataBlock[ch_idx, start:end]

# === STEP 3: CONVERT DATA TO IMAGES (every 5th frame) ===
def convert_to_images(dataBlock, output_folder, chToPlot):
    os.makedirs(output_folder, exist_ok=True)
    num_frames = dataBlock.shape[1]

    for i in range(0, num_frames, 5):
        plt.figure(figsize=(6, 3))
        for j, ch in enumerate(chToPlot):
            plt.plot(dataBlock[j, :i+1], label=f"Ch {ch}")
        plt.legend()
        plt.title(f"Frame {i}")
        plt.xlabel("Time Steps")
        plt.ylabel("Signal")
        plt.tight_layout()
        path = os.path.join(output_folder, f"frame_{i:04d}.png")
        plt.savefig(path)
        plt.close()

# === STEP 4: ASSEMBLE VIDEO ===
def create_video_from_frames(folder, video_path, fps=30):
    images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if not images:
        print("❌ No images to create video.")
        return

    first = cv2.imread(os.path.join(folder, images[0]))
    height, width, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img in images:
        frame = cv2.imread(os.path.join(folder, img))
        video.write(frame)

    video.release()
    print(f"✅ Video saved to: {video_path}")

# === STEP 5: RUN MEDIAPIPE POSE DETECTION ===
def run_mediapipe(video_path):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ MediaPipe pose detection complete.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🔄 Loading sensor data...")
    dataBlock, rate = load_data(dataFile, trial=0)

    print("✂️ Slicing data...")
    sliced = slice_data(dataBlock, chToPlot, dataTimeRange_s, rate)

    print("🖼️ Creating frames (every 5th)...")
    convert_to_images(sliced, output_dir, chToPlot)

    print("🎞️ Creating video...")
    create_video_from_frames(output_dir, video_file, fps=30)

    print("🤖 Running MediaPipe...")
    run_mediapipe(video_file)
