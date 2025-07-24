import cv2
import time

def create_Trackframes(firstframe, lastframe, *definitions):
    """
    Initialises a list with dictionary definitions 
    
    Parameters:
    firstframe: frame to start
    lastframe: frame to end 
    definitions: any dictionary items the user want to input (Will be initialised with None)

    Returns:
    track_frames: A list length of the total frames to be played with dictionary definitions
    """
    default_dict = {key: None for key in definitions} # List to store all cropped frames
    track_frames = [default_dict.copy() for _ in range(lastframe - firstframe)] # Initialize the list with the number of frames
    return track_frames

# Load the saved video
cap = cv2.VideoCapture(r"E:\STARS\StudentData\Exported_Video\annotated_output.mp4")

# Get total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = 30

start_time = 0
start_frame = int(start_time * fps) # Start frame for the clip
end_time = 30 # End time for the clip in seconds
end_frame = int(fps*end_time) #int(fCount)
frame_Index = start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_Index)
waitKeyP = 1
track_frames = create_Trackframes(start_frame, end_frame, "frame")



#print(f"Processing time: {processing_time:.2f} ms, Delay: {delay} ms")  # Optional: see timing info

while frame_Index < end_frame:
    start_time = time.time()
    i = frame_Index - start_frame #index for track_frames array
    if track_frames[i]['frame'] is None: 
        success, raw_frame = cap.read() # Returns a boolean and the next frame
        if not success: # If the frame was not read successfully, break the loop
            print("Failed to read frame")
            exit()
        track_frames[i]["frame"] = raw_frame
    else:
        raw_frame = track_frames[i]["frame"]
        

    cv2.imshow("Frame", raw_frame)

    if waitKeyP != 0:
        processing_time = (time.time() - start_time) * 1000  # in milliseconds

        print(f"{processing_time}")
        delay = max(int(1000/fps) - int(processing_time), 1)  # Ensure at least 1 ms delay
        print(f"{delay}")
        waitKeyP= delay
    key1 = cv2.waitKey(waitKeyP)

    if key1 == 32: #Space to pause
        if waitKeyP == delay:
            waitKeyP = 0
            print("Pausing") 
        else:
            frame_Index -= 1 # when we unpause we will increment, but that will skip on
            waitKeyP = delay
            print("Resuming") 
            frame_Index = frame_Index + 1
    elif key1 == 81 or key1 ==2 or key1 == ord('d'): #Left Arrow:  # Back one Frame
        waitKeyP = 0 # If we key we want to pause
        frame_Index -= 1
        if frame_Index < start_frame:
            print("Cannot go further back, press space to continue")
            frame_Index = start_frame
    elif key1 == 84 or key1 == 1 or key1 == ord('s'):  # Down Arrow Back one Second
        #print(f"back one second: {fps} frames")
        waitKeyP = 0
        frame_Index -= fps
        if frame_Index < start_frame:
            print("Cannot go further back, press space to continue")
            frame_Index = start_frame
    elif key1 == 83 or key1 == 3 or key1 == ord('g'):  #Right Arrrow Step forwared One Frame
        #print(f"Forward one frame")
        waitKeyP = 0 # If we key we want to pause
        frame_Index += 1 
        if (frame_Index - start_frame) >= len(track_frames):
            #print("Reached the end of video")
            frame_Index -= 1 
            #continue             
    elif key1 == 82 or key1 == 0 or key1 == ord('h'):  #Up Arrow Forward one second
        #print(f"forward one second: {fps} frames")
        waitKeyP = 0 # If we key we want to pause
        frame_Index += fps
        #if i >= len(track_frames):
        if track_frames[frame_Index - start_frame]['frame'] is None:
            frame_Index -= fps
            print("Reached the end of buffered video")
            #continue                   
    elif key1 == ord('q'):
        print("Quitting.")
        exit()


    # If we are not paulsed go to the next frame
    if waitKeyP != 0: frame_Index = frame_Index + 1 


