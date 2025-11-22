
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

Runthrough = False 
Playback = True 

FPS = 30  # Frames per second

modelDir = "Models"   

vidDir = r"." 


dir = r"/StudentData"  



videoInputFile = r"/video_hallwayTests/poll_run_7-10-2025_10-50-56 AM.asf" # Vib data run 1, stomp lines up with 1 sec first window

vibration_data_file = r"/vibration_test_data/Jack_clockTest_interuptVPoll.hdf5"


output_dir = f"{vidDir}/{dir}"  # Set your own output path
fileName = f"{output_dir}/{videoInputFile}"
print(f"Opening video: {fileName}")


model_path = f"{modelDir}/pose_landmarker_heavy.task" # 29.2 MiB
pose_detector = PoseDetector(model_path)

# ===== Global variables
windowLen_s = 1 #5
windowInc_s = 0.5 #1


video = VideoReader(fileName)


fCount = video.total_frames
width = video.width
height = video.height


frameTime_ms = 1000/FPS #How long of a time does each frame cover

# Fit to the display
dispFact = 2
displayRez = (int(width/dispFact), int(height/dispFact))
displayRezsquare = (int(height/dispFact), int(height/dispFact)) 

# For output (writing anotated videos)

fullFrameVidOut = VideoWriter(f"{output_dir}/annotated_output_{videoInputFile}", 
                              FPS, resolution=displayRez)

croppedVidOut = VideoWriter(f"{output_dir}/annotated_output_crop_{videoInputFile}", 
                            FPS, resolution=displayRezsquare)


#vibration properties
vib = vibDataWindow(
    dir_path=f"{output_dir}",
    #dir_path=r"E:\STARS\StudentData\25_07_10\subject_2",
    #data_file=r"Jack_clockTest_interuptVPoll.hdf5",
    data_file= vibration_data_file,
    trial_to_plot=1, #First trial is 0
    old_data=False,
    window=windowLen_s
)



# Define video writers (90-frame clip, initialized when needed)
out_full = None
out_crop = None

clip_start = 0  # Example: clip starts at frame 200
clip_length = int(fCount)  # Length of the clip in frames
clip_end = clip_start + clip_length
maintain_height_max = height
maintain_height_min = 0
maintain_width_max = width
maintain_width_min = 0


fourcc = cv2.VideoWriter_fourcc(*'XVID')


# === Set time to start/end
start_time = 0

start_frame = int(start_time * FPS) # Start frame for the clip
end_time = 30 # End time for the clip in seconds
end_frame = FPS*end_time #int(fCount)
# === saves dimensions for first crop
max_height = height
min_height = 0
max_width = width
min_width = 0
# === Initialise arrays and variables to send to functions
maintain_dim = [0,height,0,width] # Initializes maintain dimensions
smoothed_dim = [None, None, None, None]  # Initialize smoothed dimensions -> [smoothed_min_h, smoothed_max_h, smoothed_min_w, smoothed_max_w]
time_tracker = timeWith_ms(frameTime_ms) #Creates object
alpha = .1  # smoothing factor between 0 (slow) and 1 (no smoothing)
direction  = "North" #Default direction 
track_frames = create_Trackframes(start_frame, end_frame, "frame", "cropped_frame", "landmarks",
                                  "LeftToe_Dist","RightToe_Dist", "RightHeel_Dist", "LeftHeel_Dist", 
                                 "seconds_sinceMid", "toeVel", "heelVel") #array to track information about each frame
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


# === Write to file (header)
csvOutputFile = os.path.splitext(videoInputFile)[0] + ".csv"
csv_file = f"{output_dir}/{csvOutputFile}"

print(f"\nCreating CSV file at: {csv_file}\n")

with open(csv_file, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([
                "Time", "Seconds_Mid",
        "LeftHeel_Dist", "RightHeel_Dist" , 
        "LeftToe_Dist" , "RightToe_Dist",
    ]) 



# === Sets the video to specified index
frame_Index = start_frame

video.set_frame(frame_Index)

# === Begin process of cropping, saving, and playback

toeVel_mps = 0
framewith_data = 0

vibImage_rgba = None
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
    
    i = frame_Index - start_frame #index for track_frames array
    # === Reads and loads new frames in array
    if track_frames[i]['frame'] is None: 
        
        #success, raw_frame = videoOpbject.read()
        raw_frame = video.read()
        if raw_frame is None:
            print("Failed to read frame")
            break


      
        total_seconds = seconds_sinceMidnight(raw_frame, frame_Index, time_tracker) 


        # === Crops full frame. Draws the cropped area on full frame
        newDim_Frame = raw_frame[min_height:max_height,min_width:max_width,:].copy() #crops frame
        cv2.rectangle(raw_frame, (min_width,max_height), (max_width, min_height), [255,0,0], 5)
        # ===Is there a cropped frame to send to model?
        if newDim_Frame is not None: 
            good = False
        # === Returns landmarks based on person

            good, result, adjusted_time_ms = pose_detector.detect(newDim_Frame, frame_Index, frameTime_ms)

        # === 
            if good and result is not None:
                landmarks = result.pose_landmarks[0]
                constPixL_Toe, crop_prevPixL_Toe = constantSize(landmarks[31],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixL_Toe)
                drawLandmark_circle(newDim_Frame, landmarks[31], [230, 216, 173], constPixL_Toe) #left toe is light blue
                constPixL_Heel, crop_prevPixL_Heel = constantSize(landmarks[29],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixL_Heel)
                drawLandmark_circle(newDim_Frame, landmarks[29], [139, 0, 0],constPixL_Heel) # left heel is dark blue
                constPixR_Heel, crop_prevPixR_Heel = constantSize(landmarks[32],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixR_Heel)
                drawLandmark_circle(newDim_Frame, landmarks[32], [102, 102, 255],constPixR_Heel) # right toe is light red 
                constPixR_Toe, crop_prevPixR_Toe = constantSize(landmarks[30],cropped_pixel_incm, frame_Index, start_frame, end_frame, crop_prevPixR_Heel)
                drawLandmark_circle(newDim_Frame, landmarks[30], [0, 0, 139],constPixR_Toe) #right heel is dark red

                landmarks_of_fullscreen(landmarks, min_width, max_width, min_height, max_height) 
                #=== Draws landmarks and expands them according to pixel size
                
                constPixL_Toe, prevPixL_Toe = constantSize(landmarks[31],pixel_incm, frame_Index, start_frame, end_frame, prevPixL_Toe)
                drawLandmark_circle(raw_frame, landmarks[31], [230, 216, 173],constPixL_Toe) #left toe is light blue
                constPixL_Heel, prevPixL_Heel = constantSize(landmarks[29],pixel_incm, frame_Index, start_frame, end_frame, prevPixL_Heel)
                drawLandmark_circle(raw_frame, landmarks[29], [139, 0, 0],constPixL_Heel) # left heel is dark blue
                constPixR_Heel, prevPixR_Heel = constantSize(landmarks[32],pixel_incm, frame_Index, start_frame, end_frame, prevPixR_Heel)
                drawLandmark_circle(raw_frame, landmarks[32], [102, 102, 255],constPixR_Heel) # right toe is light red 
                constPixR_Toe, prevPixR_Toe = constantSize(landmarks[30],pixel_incm, frame_Index, start_frame, end_frame, prevPixR_Heel)
                drawLandmark_circle(raw_frame, landmarks[30], [0, 0, 139],constPixR_Heel) #right heel is dark red 
                # === Get new frame dimensions           
                min_width, max_width, min_height, max_height, maintain_dim  = crop_to_square(raw_frame, landmarks, direction ,maintain_dim) 
                smoothed_dim, min_width, max_width, min_height, max_height  = smooth_crop_dim(smoothed_dim, min_width, max_width, min_height, max_height) 
                # === Saves data to array
                track_frames[i]["landmarks"] = landmarks # Store the landmarks in the track_frames list
                    # === Calculates distance
                left_distHeel = find_dist_from_y(track_frames[i]["landmarks"][29].y*height)
                right_distHeel = find_dist_from_y(track_frames[i]["landmarks"][30].y*height)
                left_distToe = find_dist_from_y(track_frames[i]["landmarks"][31].y*height)
                right_distToe = find_dist_from_y(track_frames[i]["landmarks"][32].y*height)

                
                track_frames[i]["LeftToe_Dist"] = left_distToe # Store the left toe distance in the track_frames list
                track_frames[i]["RightToe_Dist"] = right_distToe # Store the left toe distance in the track_frames list
                track_frames[i]["RightHeel_Dist"] = right_distHeel
                track_frames[i]["LeftHeel_Dist"] = left_distHeel 
                
                track_frames[i]["seconds_sinceMid"] = safe_divide(i, FPS)
                # Calculate the walking speed 
                # Every n seconds (how many frames is that)
                if framewith_data >= (windowLen_s+1)*FPS:    # don't run if we don't have a windows worth of data
                                                # Also, skip the times that don't have rollovers
                    if framewith_data % (windowInc_s*FPS) == 0: # run every overlap
                        #print(f"Calculate ms at frame: {i}, FPS:{FPS}, inc: {windowInc_s} sec")
                        #print(f"distance: {track_frames[i]["LeftToe_Dist"]}, landmark: {track_frames[i]["landmarks"][29].y}")
                        heelVel_mps = calculate_avg_landMark_velocity(track_frames, left="LeftHeel_Dist", right="RightHeel_Dist", curentFrame=i, nPoints= windowLen_s*FPS, verbose=False)
                        toeVel_mps = calculate_avg_landMark_velocity(track_frames, left="LeftToe_Dist", right="RightToe_Dist", curentFrame=i, nPoints= windowLen_s*FPS, verbose=False)

                        # TODO:Jack Get vibration data

                        # send time  seconds since midnight and location of walker
                        # returns:  img_rgba = np.asarray(canvas.buffer_rgba())
                        vibImage_rgba = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[1-1])
                        vibImage_rgba2 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[2-1], colVar ='blue', debug=True)
                        vibImage_rgba3 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[3-1], colVar ='orange')
                        vibImage_rgba4 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[4-1], colVar ='darkgoldenrod')
                        vibImage_rgba5 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[5-1], colVar ='green')
                        vibImage_rgba6 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[6-1], colVar ='purple')
                        vibImage_rgba7 = vib.vib_get(time=total_seconds, distanceFromCam=50, chToPlot=[7-1], colVar ='olive')

                        


                if vibImage_rgba is not None:
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba, loc_x=950, loc_y=int(findPixfromDist(7.59)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba2, loc_x=1900, loc_y=int(findPixfromDist(10.26)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba3, loc_x=1000, loc_y=int(findPixfromDist(12.32)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba4, loc_x=1750, loc_y=int(findPixfromDist(13.99)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba5, loc_x=1050, loc_y=int(findPixfromDist(17)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba6, loc_x=1600, loc_y=int(findPixfromDist(23.32)), dim_x=200, dim_y=200) # overlay at this position
                    raw_frame = overlay_image(raw_frame.copy(), vibImage_rgba7, loc_x=1100, loc_y=int(findPixfromDist(26.82)), dim_x=200, dim_y=200) # overlay at this position
                    

                track_frames[i]["toeVel"] = toeVel_mps
                track_frames[i]["heelVel"] = toeVel_mps

                text = [
                    f"Seconds: {track_frames[i]['seconds_sinceMid']:.3f} s",
                    f"Left Heel: {track_frames[i]['LeftHeel_Dist']:.2f} m", 
                    f"Left Toe: {track_frames[i]['LeftToe_Dist']:.2f} m", 
                    f"Right Heel: {track_frames[i]['RightHeel_Dist']:.2f} m",
                    f"Right Toe: {track_frames[i]['RightToe_Dist']:.2f} m",
                    'Previous Window: ',
                    f"Toe Vel: {track_frames[i]['toeVel']:.2f} m/s",
                    f"Heel Vel: {track_frames[i]['heelVel']:.2f} m/s",
                    ]
                framewith_data +=1

                # TODO: Add vibration data to frame
            else: # not good or no result
                text = [
                    ]
                if frame_Index % 2 ==0:
                    min_width, max_width, min_height, max_height, direction = crop_to_Southhall() #, landmarks
                else:
                    min_width, max_width, min_height, max_height, direction = crop_to_Northhall() #, landmarks
            resized_rawframe = cv2.resize(raw_frame, displayRez)
            resizedframe = cv2.resize(newDim_Frame, displayRezsquare)
            # ===resize for viewing and save in array
            put_text(text,  resizedframe)
            put_text(text, resized_rawframe)
            #print(f"shape | raw_frame {raw_frame.shape}, resized_rawframe {resized_rawframe.shape}")
            track_frames[i]["frame"] = resized_rawframe
            track_frames[i]["cropped_frame"] = resizedframe
            fullFrameVidOut.write(resized_rawframe)  # Save the frame to video
            croppedVidOut.write(resizedframe)  # Save the frame to video

    else:
        resized_rawframe = track_frames[i]["frame"]
        resizedframe = track_frames[i]["cropped_frame"]

    cv2.imshow("Zoomed Frame: ", resizedframe)
    cv2.imshow("Frame", resized_rawframe)



    new_index = handle_keyboard(frame_Index, start_frame, FPS)

    if new_index is None:
        print("Quitting.")
        break

    frame_Index = new_index
    

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
        #frame_Index,
        total_seconds,  # Convert to seconds
        left_distHeel,
        right_distHeel,
        left_distToe,
        right_distToe
    ])

fullFrameVidOut.close()
croppedVidOut.close()

cv2.destroyAllWindows()