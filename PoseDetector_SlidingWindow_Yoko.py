####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# Pose tracking with MediaPipe and OCR + CSV output — Heels, Toes, Hips
####

# === Imports ===
import time
import math
import csv
import sys
import os
from collections import deque, defaultdict

import cv2
import pytesseract

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# === Fix import path to reach distance_position.py ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_position import find_dist_from_y  # Distance calculation function

# === MODEL PATH ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task"

# === VIDEO FILE ===
video_dir = '/Volumes/MY PASSPORT/SFSU_STARS/25_06_18/Subject_1'
video_file = 'Sub_1_Run_1_6-18-2025_11-45-46 AM.asf'
fileName = f"{video_dir}/{video_file}"

# === Output Directory ===
csv_path = "/Users/yokolu/NSF_Floor_Vib_Camera-Labeling/time_aligned_distances.csv"
output_dir = "/Users/yokolu/NSF_Floor_Vib_Camera-Labeling"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory: {output_dir}")

# === Define CSV and Graph Paths ===
csv_path = f"{output_dir}/heel_tracking_output.csv"
velocity_csv_path = f"{output_dir}/velocity_output.csv"
combined_csv_path = f"{output_dir}/time_aligned_distances.csv"
avg_csv_path = f"{output_dir}/avg_velocity_0_to_5s.csv"

# === Sliding Window Output ===
sliding_csv_path = f"{output_dir}/sliding_window_velocity.csv"


# === Open video ===
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = videoObject.get(cv2.CAP_PROP_FPS)
fCount = videoObject.get(cv2.CAP_PROP_FRAME_COUNT)
w = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))
dispFact = 2
displayRez = (int(w / dispFact), int(h / dispFact))

# === MediaPipe Setup ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def drawLandmark(frame, landmark, color=(0, 0, 255)):
    center = (int(landmark.x * w), int(landmark.y * h))
    cv2.circle(frame, center, 6, color, -1)

def drawLine(frame, lm1, lm2, color):
    pt1 = (int(lm1.x * w), int(lm1.y * h))
    pt2 = (int(lm2.x * w), int(lm2.y * h))
    cv2.line(frame, pt1, pt2, color, 3)

# === Frame Timing and Clip Setup ===
frameTime_ms = 1000 / fps
clipStartTime_s = 59
clip_length_s = 12 #video length in seconds
start_time = clipStartTime_s

# === CSV Setup ===
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Timestamp_ms", "Landmark", "Y_Norm", "Y_Pixels", "Distance_m"])

velocity_results = []  # [] # This will hold the final velocity results for each landmark
heel_toe_windows = {}  # {LandmarkName: deque of (time, distance)} 
                       # {} is a dictionary, where each key is a landmark name (e.g., "LeftHeel") and the value is a deque (double-ended queue) of tuples (time, distance)
sliding_results = defaultdict(list)  # For 1-second increment 5-second windows

# === Frame Loop over 5s window ===
videoObject.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
clipStartFrame = int(start_time * fps)
clipRunFrames = int(clip_length_s * fps)
print(f"\n🔁 Processing clip starting at {start_time:.2f} s")

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=True
)
landmarker = PoseLandmarker.create_from_options(options)

for i in range(clipRunFrames):
    frame_timestamp_ms = int((clipStartFrame + i) * frameTime_ms)
    clip_relative_s = i / fps  # Always from 0.00 to 5.00

    success, frame = videoObject.read()
    if not success:
        print("⚠️ Frame read failure")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if len(result.pose_landmarks) == 0:
        continue

    landmarks = result.pose_landmarks[0]

   # Loop through a dictionary that maps each landmark name (e.g., "LeftHeel")
   # to its corresponding index in MediaPipe's pose landmarks
    for name, idx in {
        "LeftHeel": 29,
        "RightHeel": 30,
        "LeftToe": 31,
        "RightToe": 32
    }.items():
        y_norm = landmarks[idx].y # Extract the normalized Y-coordinate (between 0 and 1) for the specified landmark
        y_pix = y_norm * h # Convert the normalized Y-coordinate into pixel units using the frame height
        dist_m = find_dist_from_y(y_pix, debug=False) # Estimate the real-world vertical distance (in meters) from the pixel Y-coordinate


        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                i, frame_timestamp_ms, name,
                f"{y_norm:.5f}", f"{y_pix:.2f}", f"{dist_m:.4f}"
            ])

       # If this is the first time we are seeing this landmark name (e.g., "LeftHeel"), initialize a new deque (double-ended queue)
        if name not in heel_toe_windows:
            heel_toe_windows[name] = deque()
        heel_toe_windows[name].append((clip_relative_s, dist_m))

        # Store all raw data for sliding window summary
        if 'all_distances' not in globals():
            all_distances = defaultdict(list)
        all_distances[name].append((clip_relative_s, dist_m))


                # === Compute sliding window velocity every 1 second ===
        if len(heel_toe_windows[name]) >= 2:
            t0 = heel_toe_windows[name][0][0]
            t1 = heel_toe_windows[name][-1][0]

            if t1 - t0 >= 4.9:  # At least 5 seconds window
                times = np.array([t for t, d in heel_toe_windows[name]])
                dists = np.array([d for t, d in heel_toe_windows[name]])
                A = np.vstack([times, np.ones_like(times)]).T
                m, b = np.linalg.lstsq(A, dists, rcond=None)[0]

                # Save to sliding_results
                if 'sliding_results' not in globals():
                    sliding_results = defaultdict(list)
                sliding_results[name].append((t0, t1, m))

        # Maintain the sliding window length: keep only the most recent 5 seconds of data
        # As long as the window is longer than z5 seconds, remove the oldest entry from the left
        while heel_toe_windows[name] and heel_toe_windows[name][-1][0] - heel_toe_windows[name][0][0] > 5.0:
            heel_toe_windows[name].popleft()

    # === Draw landmarks and lines ===
    drawLandmark(frame, landmarks[29], (255, 0, 0))    # Left Heel
    drawLandmark(frame, landmarks[30], (0, 0, 255))    # Right Heel
    drawLandmark(frame, landmarks[31], (255, 255, 0))  # Left Toe
    drawLandmark(frame, landmarks[32], (0, 255, 255))  # Right Toe
    drawLandmark(frame, landmarks[23], (255, 0, 255))  # Left Hip
    drawLandmark(frame, landmarks[24], (0, 255, 0))    # Right Hip

    drawLine(frame, landmarks[23], landmarks[29], (255, 0, 128))
    drawLine(frame, landmarks[29], landmarks[31], (255, 128, 0))
    drawLine(frame, landmarks[24], landmarks[30], (0, 255, 128))
    drawLine(frame, landmarks[30], landmarks[32], (0, 128, 255))

    frame = cv2.resize(frame, displayRez)
    cv2.imshow("Input", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        videoObject.release()
        cv2.destroyAllWindows()
        exit()

# === Final Least Squares Fit (Entire 5s Window) ===
print("\n📈 Final Least Squares Linear Fit and R2 for 0–5s Window:")
with open(avg_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Landmark", "Slope (m/s)", "Intercept", "R2"])

    for name in ["LeftHeel", "RightHeel", "LeftToe", "RightToe"]:
        if name in heel_toe_windows and len(heel_toe_windows[name]) >= 2: # Ensure the landmark has at least 2 data points (needed for linear regression)
            times = np.array([t for t, d in heel_toe_windows[name]]) # Extract the time values into a NumPy array
            dists = np.array([d for t, d in heel_toe_windows[name]])
            A = np.vstack([times, np.ones_like(times)]).T # Create the design matrix A for linear regression (A = [t, 1])
            m, b = np.linalg.lstsq(A, dists, rcond=None)[0] # Perform least squares fit: solve A * [m, b] = dists

            y_pred = m * times + b # Compute predicted distances using the model: y = m * t + b
            ss_res = np.sum((dists - y_pred) ** 2) # Compute sum of squared residuals (errors between actual and predicted distances)
            ss_tot = np.sum((dists - np.mean(dists)) ** 2) # Compute total sum of squares (variance of actual distances)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan') # Avoid division by zero
            velocity_results.append([name, times[0], times[-1], m]) # Store result in velocity_results list for later use or reporting

            writer.writerow([name, f"{m:.4f}", f"{b:.4f}", f"{r2:.4f}"])
            print(f"{name}: d = {m:.3f} * t + {b:.3f}, R2 = {r2:.4f}")
        else:
            print(f"{name}: Not enough data for least squares fit.")

# === Save velocity CSV ===
with open(velocity_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Landmark", "Start_s", "End_s", "Velocity_m_per_s"])
    writer.writerows(velocity_results)
print(f"✅ Velocity CSV saved: {velocity_csv_path}")

# === Save Sliding Window Velocity CSV ===
with open(sliding_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Landmark", "Window_Start_s", "Window_End_s", "Velocity_m_per_s"])
    for name, segments in sliding_results.items():
        for t0, t1, v in segments:
            writer.writerow([name, f"{t0:.2f}", f"{t1:.2f}", f"{v:.4f}"])
print(f"✅ Sliding window velocity CSV saved: {sliding_csv_path}")

# === Plot Sliding Window Velocities ===
for name, segments in sliding_results.items():
    times = [(t0 + t1) / 2 for t0, t1, _ in segments]
    velocities = [v for _, _, v in segments]

    plt.figure()
    plt.plot(times, velocities, marker='o', linestyle='-', label=f"{name} Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"{name} — Sliding Velocity (5s window, 1s step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = Path(output_dir) / f"{name}_sliding_velocity_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved sliding velocity plot: {plot_path}")
    
# === Time-aligned Distance CSV ===
all_times = set()
for values in heel_toe_windows.values():
    all_times.update([round(t, 3) for t, _ in values])
all_times = sorted(all_times)

distance_lookup = {
    name: dict((round(t, 3), d) for t, d in values) # For each landmark, map time (rounded to 3 decimals) to its distance
    for name, values in heel_toe_windows.items() # Iterate through all landmark names and their recorded values
}
landmark_names = ["LeftHeel", "RightHeel", "LeftToe", "RightToe"]

with open(combined_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Time_s"] + [f"{name}_Dist_m" for name in landmark_names])
    for t in all_times:
        row = [f"{t:.3f}"]
        for name in landmark_names:
            val = distance_lookup.get(name, {}).get(t, "")
            row.append(f"{val:.4f}" if val != "" else "")
        writer.writerow(row)
print(f"✅ Combined CSV saved: {combined_csv_path}")

# === Average Velocity Report ===
print("\n📊 Average Velocities (m/s):")
velocity_by_landmark = defaultdict(list) #dictionary from collections module.
# This will hold lists of velocities for each landmark name
# Iterate through the velocity results and group by landmark name  
for name, t0, t1, v in velocity_results:
    velocity_by_landmark[name].append(v)
for name, velocities in velocity_by_landmark.items():
    avg = sum(velocities) / len(velocities)
    print(f"{name}: {avg:.3f}")


# === Detailed Sliding Window Summary with Least Squares ===
print("\n📈 Sliding Window Velocity Summary with Least Squares (5s window, 1s steps):")
# === Sliding Window Analysis ===
window_size = 5.0
step_size = 1.0
results = []


for name in all_distances:
    data = all_distances[name]
    data.sort()  # ✅ Now sorting will work
    
    times_all = np.array([t for t, _ in data])
    start = times_all[0]
    end = times_all[-1]

    while start + window_size <= end:
        t0 = round(start, 1)
        t1 = round(start + window_size, 1)
        window_data = [(t, d) for (t, d) in data if t0 <= t <= t1]
        if len(window_data) < 2:
            start += step_size
            continue
        times = np.array([t for t, _ in window_data])
        dists = np.array([d for _, d in window_data])
        A = np.vstack([times, np.ones_like(times)]).T
        m, b = np.linalg.lstsq(A, dists, rcond=None)[0]
        y_pred = m * times + b
        ss_res = np.sum((dists - y_pred) ** 2)
        ss_tot = np.sum((dists - np.mean(dists)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        results.append([name, t0, t1, round(m, 4), round(b, 4), round(r2, 4)])
        start += step_size

# Create the df_r2 DataFrame from the results
df_r2 = pd.DataFrame(results, columns=["Landmark", "Start_s", "End_s", "Velocity_m_per_s", "Intercept", "R2"])

grouped = df_r2.groupby("Landmark")
seen_windows = set()

for name, group in grouped:
    print(f"\n▶ {name} Sliding Velocities:")
    for _, row in group.iterrows():
        t0_int = int(row["Start_s"])
        t1_int = int(row["End_s"])
        if t1_int - t0_int == 5 and (name, t0_int, t1_int) not in seen_windows:
            slope = row["Velocity_m_per_s"]
            intercept = row["Intercept"]
            r2 = row["R2"]
            print(f"{t0_int}–{t1_int} seconds: "
                  f"d = {slope:.3f} * t + {intercept:.3f}, R² = {r2:.4f}")
            seen_windows.add((name, t0_int, t1_int))


# === Cleanup ===
videoObject.release()
cv2.destroyAllWindows()

# === File Path ===
csv_path = "/Users/yokolu/NSF_Floor_Vib_Camera-Labeling/time_aligned_distances.csv"
output_dir = Path("/Users/yokolu/NSF_Floor_Vib_Camera-Labeling")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(csv_path)
time_col = "Time_s"
landmarks = ["LeftHeel_Dist_m", "RightHeel_Dist_m", "LeftToe_Dist_m", "RightToe_Dist_m"]

# === Plot Each Landmark ===
for landmark in landmarks:
    times = df[time_col].astype(float)
    dists = pd.to_numeric(df[landmark], errors='coerce')

    # Drop NaN values (where distance is missing)
    valid = ~np.isnan(dists)  # Create a boolean mask for non-NaN (valid) values
    times = times[valid]  # Filter out times that correspond to NaN distances
    dists = dists[valid]  # Keep only valid distances

    if len(times) < 2:  # If there's not enough data to fit or plot, skip this landmark
        print(f"⚠️ Not enough data to plot {landmark}")
        continue

    # === Linear Regression (Least Squares) ===
    A = np.vstack([times, np.ones_like(times)]).T
    m, b = np.linalg.lstsq(A, dists, rcond=None)[0]
    y_pred = m * times + b

    # === R² Calculation ===
    ss_res = np.sum((dists - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((dists - np.mean(dists)) ** 2)  # Total sum of squares
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')  # R² = 1 - (SSres/SStot), avoid division by zero

    # === Plot ===
    plt.figure()
    plt.plot(times, dists, '-', color='blue', label=f'{landmark} Distance')  # Line connecting the dots
    plt.scatter(times, dists, s=5, color='blue')  

    #plt.plot(times, y_pred, '--', color='red', label=f'Linear Fit: d={m:.3f}t + {b:.3f}\nR²={r2:.4f}')

    # Plot original data and trendline
    plt.plot(times, dists, 'o', color='blue', label='Landmark Detection', markersize=3)
    plt.plot(times, y_pred, '--', color='red', label='Linear Fit')

    # Add equation and R² text beside the line
    mid_idx = len(times) // 2
    mid_x = times[mid_idx]
    mid_y = y_pred[mid_idx]
    equation_text = f"d = {m:.3f}·t + {b:.3f}\nR² = {r2:.4f}"
    plt.text(mid_x + 0.8, mid_y, equation_text, fontsize=10, color='red', verticalalignment='bottom')

    # Plot formatting
    plt.title(f"{landmark.replace('_Dist_m', '')} Distance Over Time, Subject 1, Run 1, N-S, June 18th, 2025")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.legend()

    # === Save Plot ===
    output_path = output_dir / f"{landmark.replace('_Dist_m', '')}_distance_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

results = []
window_size = 5.0
step_size = 1.0

for name in all_distances:
    data = all_distances[name]
    
    # Get min and max time across full video (not just recent 5s)
    times_all = np.array([t for t, _ in data])
    start = 0.0  # Explicitly start at 0
    end = 12.0   # Full duration of the clip

    while start + window_size <= end:
        t0 = round(start, 1)
        t1 = round(start + window_size, 1)

        # Select points in current window
        window_data = [(t, d) for (t, d) in data if t0 <= t <= t1]
        if len(window_data) < 2:
            start += step_size
            continue

        times = np.array([t for t, _ in window_data])
        dists = np.array([d for _, d in window_data])
        A = np.vstack([times, np.ones_like(times)]).T
        m, b = np.linalg.lstsq(A, dists, rcond=None)[0]
        y_pred = m * times + b
        ss_res = np.sum((dists - y_pred) ** 2)
        ss_tot = np.sum((dists - np.mean(dists)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        results.append([name, int(t0), int(t1), round(m, 4), round(b, 4), round(r2, 4)])
        start += step_size


# Display and Print the Results
df_r2 = pd.DataFrame(results, columns=["Landmark", "Start_s", "End_s", "Velocity_m_per_s", "Intercept", "R2"])

for name in ["LeftHeel", "RightHeel", "LeftToe", "RightToe"]:
    group = df_r2[df_r2["Landmark"] == name]
    print(f"\n▶ {name} Sliding Velocities:")
    for _, row in group.iterrows():
        print(f"{row['Start_s']}–{row['End_s']} seconds: {row['Velocity_m_per_s']:.3f} m/s (R² = {row['R2']:.4f})")

# === Plot each 5-second window separately with fit line and R² ===
output_dir = Path("/Users/yokolu/NSF_Floor_Vib_Camera-Labeling-CSV")
output_dir.mkdir(parents=True, exist_ok=True)

for idx, row in df_r2.iterrows():
    name = row["Landmark"]
    t0 = row["Start_s"]
    t1 = row["End_s"]
    v = row["Velocity_m_per_s"]
    b = row["Intercept"]
    r2 = row["R2"]

    # Get the original distance data from the global `all_distances`
    raw_data = all_distances.get(name, [])
    window_data = [(t, d) for (t, d) in raw_data if t0 <= t <= t1]

    if len(window_data) < 2:
        continue

    times = np.array([t for t, _ in window_data])
    dists = np.array([d for _, d in window_data])
    fit_line = v * times + b

    # === Plot of equation and R2 label beside the trendline ===
    plt.figure()
    plt.plot(times, dists, 'o', label="Landmark detection", markersize=4)
    plt.plot(times, fit_line, '--r', label=f"Fit Line")

    # Find mid-point of the line for placing the text
    mid_idx = len(times) // 2
    mid_x = times[mid_idx]
    mid_y = fit_line[mid_idx]

    # Create label text
    equation_text = f"d = {m:.3f}·t + {b:.3f}\nR² = {r2:.4f}"

    # Plot text near the center of the line
    plt.text(mid_x + 0.8, mid_y, equation_text, fontsize=10, color='red', verticalalignment='bottom')

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title(f"{name}: {t0}–{t1} sec")
    plt.grid(True)
    plt.legend()

    plot_path = output_dir / f"{name}_{int(t0)}-{int(t1)}_SlidingWindow.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved plot: {plot_path}")

# === Save Excel Sheet ===
excel_output_path = Path("/Users/yokolu/NSF_Floor_Vib_Camera-Labeling-CSV/SlidingWindowVelocities.xlsx")
excel_output_path.parent.mkdir(parents=True, exist_ok=True)
df_r2.to_excel(excel_output_path, index=False)
print(f"✅ Excel sheet saved: {excel_output_path}")

# === Plot for each Landmark ===
for name, group in df_r2.groupby("Landmark"):
    times = [(start + end) / 2 for start, end in zip(group["Start_s"], group["End_s"])]
    velocities = group["Velocity_m_per_s"]

    # Least squares for trend line
    A = np.vstack([times, np.ones_like(times)]).T
    m, b = np.linalg.lstsq(A, velocities, rcond=None)[0]
    fit_line = m * np.array(times) + b
    r2 = 1 - np.sum((velocities - fit_line)**2) / np.sum((velocities - np.mean(velocities))**2)

    # Plot
    plt.figure()
    plt.plot(times, velocities, 'o', label="Sliding Window Velocities")
    plt.plot(times, fit_line, '--r', label=f"Fit Line")
    plt.title(f"{name} — Velocity over Time\nSliding 5s Window (1s Steps)")
    plt.xlabel("Midpoint Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()


# Load velocity data
csv_file = Path(sliding_csv_path)
df_r2 = pd.read_csv(csv_file)

# Show columns for debugging
print("📋 Columns in CSV:", df_r2.columns.tolist())

# Normalize alternate column names
df_r2.rename(columns={
    "Window_Start_s": "Start_s",
    "Window_End_s": "End_s"
}, inplace=True)


# Group by window (Start_s and End_s)
window_dict = {}
for _, row in df_r2.iterrows():
    key = (row["Start_s"], row["End_s"])
    if key not in window_dict:
        window_dict[key] = {}
    window_dict[key][row["Landmark"]] = row["Velocity_m_per_s"]

# Output directory for plots
output_dir = Path("/Users/yokolu/NSF_Floor_Vib_Camera-Labeling-CSV")
plot_dir = output_dir / "HeelToe_Avg_PerWindow_Plots"
plot_dir.mkdir(parents=True, exist_ok=True)

heel_results = []
toe_results = []

for (start, end), landmarks in sorted(window_dict.items()):
    heel_vals = [v for k, v in landmarks.items() if k in ["LeftHeel", "RightHeel"]]
    toe_vals = [v for k, v in landmarks.items() if k in ["LeftToe", "RightToe"]]

    if len(heel_vals) > 0:
        heel_avg = np.mean(heel_vals)
        heel_results.append([(start + end)/2, heel_avg])

    if len(toe_vals) > 0:
        toe_avg = np.mean(toe_vals)
        toe_results.append([(start + end)/2, toe_avg])


# === Averaged Heel/Toe Sliding Window Scatter Plots with Trendline ===
print("\n📊 Creating averaged Heel and Toe scatter plots with trendlines...")

# Collect averaged distances across Left/Right for each window
output_dir = Path("/Users/yokolu/NSF_Floor_Vib_Camera-Labeling-CSV")

# Helper fun
def average_landmarks(data1, data2):
    times = []
    distances = []
    for (t1, d1), (t2, d2) in zip(data1, data2):
        if abs(t1 - t2) < 1e-3:  # match timestamps
            times.append((t1 + t2) / 2)
            distances.append((d1 + d2) / 2)
    return list(zip(times, distances))

def get_window_data(avg_data, t0, t1):
    return [(t, d) for (t, d) in avg_data if t0 <= t <= t1]

# Prepare averaged Toe and Heel data
avg_heel_data = average_landmarks(all_distances["LeftHeel"], all_distances["RightHeel"])
avg_toe_data = average_landmarks(all_distances["LeftToe"], all_distances["RightToe"])

# Loop through sliding 5s windows with 1s step
window_size = 5.0
step_size = 1.0
clip_duration = 12.0
start = 0.0

while start + window_size <= clip_duration:
    t0 = round(start, 1)
    t1 = round(start + window_size, 1)

    for label, data in [("Heel", avg_heel_data), ("Toe", avg_toe_data)]:
        window_data = get_window_data(data, t0, t1)
        if len(window_data) < 2:
            continue

        times = np.array([t for t, _ in window_data])
        dists = np.array([d for _, d in window_data])
        A = np.vstack([times, np.ones_like(times)]).T
        m, b = np.linalg.lstsq(A, dists, rcond=None)[0]
        fit_line = m * times + b
        y_pred = fit_line
        r2 = 1 - np.sum((dists - y_pred) ** 2) / np.sum((dists - np.mean(dists)) ** 2)

        # === Plot of equation and R2 label beside the trendline ===
        plt.figure()
        plt.plot(times, dists, 'o', label=f"{label} Landmark Detection", color='blue', markersize=2)
        plt.plot(times, fit_line, '--', color='red', label="Fit Line")

        # Find mid-point of the line for placing the text
        mid_idx = len(times) // 2
        mid_x = times[mid_idx]
        mid_y = fit_line[mid_idx]

        # Create label text
        equation_text = f"d = {m:.3f}·t + {b:.3f}\nR² = {r2:.4f}"

        # Plot text near the center of the line
        plt.text(mid_x + 0.8, mid_y, equation_text, fontsize=10, color='red', verticalalignment='bottom')

        # Plot formatting
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title(f"{label} Avg: {int(t0)}–{int(t1)} sec")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save
        plot_path = output_dir / f"Avg{label}_{int(t0)}-{int(t1)}_SlidingWindow.png"
        plt.savefig(plot_path)
        plt.close()

    start += step_size

