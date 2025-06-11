####
#   Yoko Lu
#   STARS Summer 2025
#   Dr J Lab
###
# MediaPipe Pose Tracking with OCR Timestamp (Clean Output)
####

#Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker


import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pytesseract

# === MODEL PATH ===
model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_full.task" #9.4MB
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_heavy.task" #30.7MB
#model_path = "/Users/yokolu/Desktop/mediapipe_models/pose_landmarker_lite.task" #5.8B

#Pose detector: Input shape: 224 x 224 x 3; Pose landmarker: 256 x 256 x 3


# === CONFIGURATION ===
video_dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
video_file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
fileName = f"{video_dir}/{video_file}"

# === OPEN VIDEO ===qqq
videoObject = cv2.VideoCapture(fileName)
if not videoObject.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = videoObject.get(cv2.CAP_PROP_FPS)
fCount = int(videoObject.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(videoObject.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(videoObject.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameTime_ms = 1000 / fps

camera = cv2.VideoCapture(0)  # Webcam
pan = camera.get(cv2.CAP_PROP_PAN)
tilt = camera.get(cv2.CAP_PROP_TILT)
zoom = camera.get(cv2.CAP_PROP_ZOOM)
print(f"Pan: {pan}, Tilt: {tilt}, Zoom: {zoom}")

camera.set(cv2.CAP_PROP_PAN, 30)    # Set pan angle
camera.set(cv2.CAP_PROP_TILT, -10)  # Set tilt angle
camera.set(cv2.CAP_PROP_ZOOM, 2.0)  # Set zoom level


# === DISPLAY SETTINGS (optional) ===
dispFact = 2
displayRez = (w // dispFact, h // dispFact)

# === MediaPipe Pose Setup ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)

#Check the video, looking at the person only

cap = cv2.VideoCapture("video_file")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Get center x, y from hip landmarks
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        cx = int((left_hip.x + right_hip.x) / 2 * w)
        cy = int((left_hip.y + right_hip.y) / 2 * h)

        # Set crop size (you can adjust this)
        crop_w, crop_h = 300, 400

        # Compute top-left corner
        x1 = max(cx - crop_w // 2, 0)
        y1 = max(cy - crop_h // 2, 0)
        x2 = min(x1 + crop_w, w)
        y2 = min(y1 + crop_h, h)

        # Crop frame
        cropped_frame = frame[y1:y2, x1:x2]
    else:
        cropped_frame = frame  # fallback

    # Show cropped frame
    cv2.imshow("Centered on Person", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
        return f"{date_str} {time_str}", conf
    except:
        return "OCR Error", 0
    
#Determine the landmarks in x, y, z, and visibility
def landmarks_image(image, result):
    if not result.pose_landmarks:
        return image
#iterates over each landmark, which is a normalized unit between 0 and 1
    for idx, landmark in enumerate(result.pose_landmarks[0]):
        x = landmark.x
        y = landmark.y
        z = landmark.z
        visibility = landmark.visibility
        print(f"Landmark {idx}: x={x:.3f}, y={y:.3f}, z={z:.3f}, visibility={visibility:.2f}")

# === FRAME RANGE ===
startSecs = 125
endSecs = 155
start_frame = int(fps * startSecs)
end_frame = int(fps * endSecs)
num_frames = end_frame - start_frame

if num_frames < 0:
    print(f"⚠️ Invalid frame range.")
    exit()

# === MAIN LOOP ===
videoObject.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for i in range(num_frames):
    frame_timestamp_ms = int((start_frame + i) * frameTime_ms)
    success, frame = videoObject.read()
    
    if not success:
        print("❌ Failure reading frame")
        exit()
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    #annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    print(f"Landmarks: {result.pose_landmarks}")



    # OCR timestamp from frame
    timestamp_text, conf = getDateTime(frame)
    print(f"📅 Date: {timestamp_text}, conf: {conf}")

    # === Display Frame (optional) ===
    frame_small = cv2.resize(frame, displayRez)
    cv2.imshow("Video Frame Only", frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔴 Quit by user.")


# getDateTime
# if len(pose_landmarker_result.post_landmarks[0])
# landmarks = pose_landmarker_result.pose_landmarks[0]
# drawlandmark(frame, landmarks[0], [0,0,0] #nose) [3 times]
#landmadrk[29]: 255,0,0 left heel
#30 0,255,0 right heel
#...if len...pose_landmark_world
#NEXT
#landmarks_w = pose_landmaker_result.pose_world_landmarks[0]
#dX = 0 # landmarks_w[20].x - 30.x#
#y
#z
#math.sqrt=math(pow.dX,2) + Y + Z
#print m, ft Stride length

#show the frame --> this comes before frame = cv2.resize
#draw circle, or line, or rectangle
#cv2.circle(200,100),20,[0,0,100],1)
# if len... (the first line here) comes after pose landmaker result, before get date time, before if no success, frame read failure

#if pose_landmark_result.segmentation_mask is not None:
#Mask=np.array(pose_landmaker_result.segmentation_masks)
#mask = (mask * 255).asktype(np.uint8) #was 0.0 - 1.0 --> 0 - 255
#cv2.imshow("Seg mark", mask)

# try to change 0 to 1 at Key at the bottom, last two