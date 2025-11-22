
#Third party
import cv2 # opencv-python
import pytesseract # pip install pytesseract
import matplotlib as plt # matplotlib
import numpy as np # numpy
import csv

# Media Pipe
import mediapipe as mp  # pip install mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys 
import os

# Our stuff
from Scripts.velocity import calculate_avg_landMark_velocity 
from Scripts.cv2Utils import *
from Scripts.distance_position import find_dist_from_y 
from Scripts.OCR_Detect import timeWith_ms # Import the timeWith_ms class from OCR_Detect.py
from Scripts.vibDataChunker import vibDataWindow

from Scripts.video_io import VideoReader, VideoWriter, handle_keyboard
from Scripts.mediapipe_wrapper import PoseDetector


# ===============================
# Runtime Flags
# ===============================
Runthrough = False
Playback = True

FPS = 30  # Frames per second


# ===============================
# File / Directory Configuration
# ===============================
modelDir = "Models"
vidDir = r"."
dir = r"/StudentData"

videoInputFile = r"/video_hallwayTests/poll_run_7-10-2025_10-50-56 AM.asf"
vibration_data_file = r"/vibration_test_data/Jack_clockTest_interuptVPoll.hdf5"

output_dir = f"{vidDir}/{dir}"
fileName = f"{output_dir}/{videoInputFile}"

print(f"Opening video: {fileName}")

# ===============================
# Pose Model Initialization
# ===============================
model_path = f"{modelDir}/pose_landmarker_heavy.task"
pose_detector = PoseDetector(model_path)


# ===============================
# Window Lengths
# ===============================
windowLen_s = 1
windowInc_s = 0.5


# ===============================
# Video Reader Initialization
# ===============================
video = VideoReader(fileName)

fCount = video.total_frames
width = video.width
height = video.height

frameTime_ms = 1000 / FPS


# ===============================
# Display Resolutions
# ===============================
dispFact = 2
displayRez = (int(width / dispFact), int(height / dispFact))
displayRezsquare = (int(height / dispFact), int(height / dispFact))


# ===============================
# Output Video Writers
# ===============================
fullFrameVidOut = VideoWriter(
    f"{output_dir}/annotated_output_{videoInputFile}",
    FPS,
    resolution=displayRez
)


# ===============================
# Vibration Data Initialization
# ===============================
vib = vibDataWindow(
    dir_path=f"{output_dir}",
    data_file=vibration_data_file,
    trial_to_plot=1,
    old_data=False,
    window=windowLen_s
)


# ===============================
# Clip & Crop State Variables
# ===============================
out_full = None
out_crop = None

clip_start = 0
clip_length = int(fCount)
clip_end = clip_start + clip_length

maintain_height_max = height
maintain_height_min = 0
maintain_width_max = width
maintain_width_min = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')


# ===============================
# Playback Time Setup
# ===============================
start_time = 0
start_frame = int(start_time * FPS)
end_time = 30
end_frame = FPS * end_time

# Initial crop dims
max_height = height
min_height = 0
max_width = width
min_width = 0

# Store evolving crop dims
maintain_dim = [0, height, 0, width]
smoothed_dim = [None, None, None, None]

time_tracker = timeWith_ms(frameTime_ms)

alpha = .1
direction = "North"


# ===============================
# Tracking Arrays & Pixel Settings
# ===============================
track_frames = create_Trackframes(
    start_frame, end_frame,
    "frame", "cropped_frame", "landmarks",
    "LeftToe_Dist", "RightToe_Dist",
    "RightHeel_Dist", "LeftHeel_Dist",
    "seconds_sinceMid", "toeVel", "heelVel"
)

crop_prevPixR_Toe = None
crop_prevPixL_Toe = None
crop_prevPixR_Heel = None
crop_prevPixL_Heel = None
prevPixR_Toe = None
prevPixL_Toe = None
prevPixR_Heel = None
prevPixL_Heel = None

pixel_incm = 6
cropped_pixel_incm = 3


# ===============================
# CSV Output Initialization
# ===============================
csvOutputFile = os.path.splitext(videoInputFile)[0] + ".csv"
csv_file = f"{output_dir}/{csvOutputFile}"
csv_rows = []

# ===============================
# Set Starting Frame
# ===============================
frame_Index = start_frame
video.set_frame(frame_Index)


# ===============================
# Pre-loop Runtime Variables
# ===============================
toeVel_mps = 0
framewith_data = 0
vibImage_rgba = None

windowName = "Main Frame:"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

raw_frame = video.read()
if raw_frame is None:
    print("Failed to read frame")
    exit()

initial_seconds = seconds_sinceMidnight(raw_frame, frame_Index, time_tracker)
print(f"Initial seconds: {initial_seconds}")

video.set_frame(frame_Index)

print(f'\nProcessing frames {start_frame} to {end_frame}...\n')

left_distHeel = right_distHeel = left_distToe = right_distToe = 0


# === Sets the video to specified index
frame_Index = start_frame

video.set_frame(frame_Index)

# === Begin process of cropping, saving, and playback

toeVel_mps = 0
framewith_data = 0

vibration_colors = ['red', 'blue', 'orange', 'darkgoldenrod', 'green', 'purple', 'olive']
vibImage_rgba = [None for _ in range(7)]

windowName = "Main Frame:"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

raw_frame = video.read()
if raw_frame is None:
    print("Failed to read frame")
    exit()

initial_seconds = seconds_sinceMidnight( raw_frame, frame_Index, time_tracker)

print(f"Initial seconds: {initial_seconds}")


video.set_frame(frame_Index)

print(f'\nProcessing frames {start_frame} to {end_frame}...\n')
left_distHeel = right_distHeel = left_distToe = right_distToe = 0



while frame_Index < end_frame:
    
    i = frame_Index - start_frame  # index for track_frames array

    # ============================================================
    # === CALCULATIONS (NO DRAWING EXCEPT CROPPED-LANDMARK DRAWS)
    # ============================================================

    if track_frames[i]['frame'] is None:

        # Read frame
        raw_frame = video.read()
        if raw_frame is None:
            print("Failed to read frame")
            break

        total_seconds = seconds_sinceMidnight(raw_frame, frame_Index, time_tracker)

        # Compute current crop (old dims)
        newDim_Frame = raw_frame[min_height:max_height, min_width:max_width, :].copy()

        if newDim_Frame is not None:
            good, result, adjusted_time_ms = pose_detector.detect(
                newDim_Frame, frame_Index, frameTime_ms
            )
        else:
            good, result = False, None

        if good and result is not None:

            landmarks = result.pose_landmarks[0]


            # -------------------------------------------------------
            # Transform to fullscreen coordinates
            # (must occur AFTER cropped drawing)
            # -------------------------------------------------------
            landmarks_of_fullscreen(
                landmarks,
                min_width, max_width,
                min_height, max_height
            )

            # -------------------------------------------------------
            # Now update crop dims AFTER we used the old ones
            # -------------------------------------------------------
            min_width, max_width, min_height, max_height, maintain_dim = crop_to_square(
                raw_frame, landmarks, direction, maintain_dim
            )

            smoothed_dim, min_width, max_width, min_height, max_height = smooth_crop_dim(
                smoothed_dim, min_width, max_width, min_height, max_height
            )

            # Save landmarks
            track_frames[i]["landmarks"] = landmarks

            # Compute distances
            left_distHeel  = find_dist_from_y(landmarks[29].y * height)
            right_distHeel = find_dist_from_y(landmarks[30].y * height)
            left_distToe   = find_dist_from_y(landmarks[31].y * height)
            right_distToe  = find_dist_from_y(landmarks[32].y * height)

            track_frames[i]["LeftToe_Dist"]  = left_distToe
            track_frames[i]["RightToe_Dist"] = right_distToe
            track_frames[i]["RightHeel_Dist"] = right_distHeel
            track_frames[i]["LeftHeel_Dist"]  = left_distHeel

            track_frames[i]["seconds_sinceMid"] = safe_divide(i, FPS)

            # Velocities
            if framewith_data >= (windowLen_s + 1) * FPS:
                if framewith_data % (windowInc_s * FPS) == 0:

                    heelVel_mps = calculate_avg_landMark_velocity(
                        track_frames, left="LeftHeel_Dist", right="RightHeel_Dist",
                        curentFrame=i, nPoints=windowLen_s * FPS, verbose=False
                    )
                    toeVel_mps = calculate_avg_landMark_velocity(
                        track_frames, left="LeftToe_Dist", right="RightToe_Dist",
                        curentFrame=i, nPoints=windowLen_s * FPS, verbose=False
                    )
                    
                    
                    # Generate vibration overlays
                    for idx in range(7):
                        vibImage_rgba[idx] = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[idx], colVar=vibration_colors[idx])



            track_frames[i]["toeVel"]  = toeVel_mps
            track_frames[i]["heelVel"] = toeVel_mps

       
            framewith_data += 1

        else:
            text = []
            if frame_Index % 2 == 0:
                min_width, max_width, min_height, max_height, direction = crop_to_Southhall()
            else:
                min_width, max_width, min_height, max_height, direction = crop_to_Northhall()

    # ============================================================
    # === DRAWING SECTION (CLEAN, ORDER-FIXED) ====================
    # ============================================================

    if track_frames[i]['frame'] is None:
        draw_box(raw_frame, (min_width, max_height), (max_width, min_height))

        # Draw FULL-frame landmarks NOW (after crop dims changed)
        if good and result is not None:
            draw_feet_points(raw_frame, landmarks)

        # Vibration overlays
        if good and result is not None and vibImage_rgba is not None:
            raw_frame = draw_vibration_overlay(raw_frame.copy(), vibImage_rgba)


        # Resize + overlay text
        resized_rawframe = cv2.resize(raw_frame, displayRez)
  


        side_bar_text(resized_rawframe, track_frames[i])

        # Save output frames
        track_frames[i]["frame"] = resized_rawframe


        fullFrameVidOut.write(resized_rawframe)


    else:
        # Use previous stored frames
        resized_rawframe = track_frames[i]["frame"]


    # ============================================================
    # === DISPLAY + KEYBOARD INPUT ================================
    # ============================================================

    cv2.imshow("Frame", resized_rawframe)

    new_index = handle_keyboard(frame_Index, start_frame, FPS)
    if new_index is None:
        print("Quitting.")
        break

    frame_Index = new_index

    # ============================================================
    # === CSV OUTPUT ==============================================
    # ============================================================

    csv_rows.append([
        total_seconds,
        left_distHeel,
        right_distHeel,
        left_distToe,
        right_distToe
    ])

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time", "LeftHeel_Dist", "RightHeel_Dist",
        "LeftToe_Dist", "RightToe_Dist"
    ])
    writer.writerows(csv_rows)

fullFrameVidOut.close()

cv2.destroyAllWindows()