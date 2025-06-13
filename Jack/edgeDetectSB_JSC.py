####
#   Jack Capito
#   STARS Summer 2025
#   Dr J Lab
###
# Edgetracing experiment
####

import cv2 # pip install opencv-python
import time
import numpy as np
import csv

dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_06_03\Subject_2'
file = 's2_B8A44FC4B25F_6-3-2025_4-00-20 PM.asf'
filename = f"{dir}/{file}" # Path to the video file
print(filename)


videoObject = cv2.VideoCapture(filename) # Open the video file
if not videoObject.isOpened():
    print("Error: Could not open video file.")
    exit()
print(f"Loaded: {filename}")


# Function to get video parameters  
def perameters(videoPeram, text):
    output = videoObject.get(videoPeram)
    print(f"Loaded: {output}{text}")
    return output

# Function to filter and merge close points
def merge_close_y(points, threshold=8):
    sorted_points = sorted(points, key=lambda p: p[1])
    
    merged_points = []
    group = [sorted_points[0]]
    group_start_y = sorted_points[0][1]

    for point in sorted_points[1:]:
        if abs(point[1] - group_start_y) <= threshold:
            group.append(point)
        else:
            avg_x = sum(p[0] for p in group) / len(group)
            avg_y = sum(p[1] for p in group) / len(group)
            merged_points.append((avg_x, avg_y))
            group = [point]
            group_start_y = point[1]
    
    # Handle the final group
    if group:
        avg_x = sum(p[0] for p in group) / len(group)
        avg_y = sum(p[1] for p in group) / len(group)
        merged_points.append((avg_x, avg_y))

    return merged_points

def get_avg_points_from_lines(lines, line_mask):
    avg_points = []
    for line in lines:
        x1l, y1l, x2l, y2l = line[0]
        if abs(y2l - y1l) < 5:  # horizontal filtering
            x_avg = (x1l + x2l) // 2
            y_avg = (y1l + y2l) // 2
            avg_points.append((x_avg, y_avg))
            cv2.line(line_mask, (x1l, y1l), (x2l, y2l), (0,255,0), 2)
            # Optionally: cv2.circle(line_mask, (x_avg, y_avg), 5, (0,0,255), -1)
    return avg_points

def y_to_distance(y):
    return 2550 * np.exp(-0.0386 * y)

def y_to_distance_for_points(filtered_points):
    for point in filtered_points:
        y = point[1]
        distance = np.log(y/2550) / -0.0386  # Calculate distance from y
        print(f"x={point[0]:.2f}, y={y:.2f}, distance={distance:.2f} ft")

def exponential_fit(x_vals, y_vals):
    x = np.array(distances)
    y = np.array(y_values)

    # Transform y to ln(y)
    ln_y = np.log(y)

    # Fit ln_y = b*x + ln(a) using polyfit (degree 1)
    b, ln_a = np.polyfit(x, ln_y, 1)

    a = np.exp(ln_a)

    # Return the exponential equation as a string
    equation = f"y = {a:.4f} * e^({b:.4f}x)"
    return equation
    


#establish video perameters here
fps = perameters(cv2.CAP_PROP_FPS, "fps")
vidHeight = perameters(cv2.CAP_PROP_FRAME_HEIGHT, "px height")
vidWidth = perameters(cv2.CAP_PROP_FRAME_WIDTH, "px width")
frameCount = perameters(cv2.CAP_PROP_FRAME_COUNT, " frames in frameCount")
dispFact=2 #how much you're reducing the output resolution by

# Calculate the display resolution by dividing the width and height by the display factor
displayRez = (int(vidWidth/dispFact), int(vidHeight/dispFact)) 


##video player
#video start point
vidStartFrame = 2000
videoObject.set(cv2.CAP_PROP_POS_FRAMES, int(vidStartFrame))


for i in range(int(frameCount)): 
    # how far into the video you are printout
    print(f"load frame {i+vidStartFrame}")
    start_time = time.time()  # Start timing

    success, frame = videoObject.read() # .read() returns a boolean value and the frame itself. 
                                        #success (t/f): did the frame read successfull?
                                        #frame: the video image
    #check to see if we actually loaded a frame
    if not success:
        print(f"frame read failure")
        exit()


    ## Do the edge tracking
    # Define your crop region (adjust as needed)
    x1, x2 = 1300, 1650
    y1, y2 = 0, int(vidHeight)

    # Video processing to find the lines
    cropped_frame = frame[y1:y2, x1:x2, :]  # Crop the color image
    blackWhite = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(blackWhite, 80, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)

    # Create a blank mask for lines (same size as crop, 3 channels)
    line_mask = np.zeros_like(cropped_frame)

    # Calculate center points for each line
    avg_points = get_avg_points_from_lines(lines, line_mask)

    # Filter and merge close points based on y-coordinates
    filtered_points = merge_close_y(avg_points)


    # Draw the filtered points (ensure integer coordinates)
    for (x, y) in filtered_points:
        cv2.circle(line_mask, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red dot for merged midpoint

    #print(f"Filtered points: {filtered_points}")  # Show the filtered points in the terminal

    y_to_distance_for_points(filtered_points) 

    # Place the mask back into the original frame size
    overlay_mask = np.zeros_like(frame)
    overlay_mask[y1:y2, x1:x2] = line_mask

    # Overlay the mask onto the original frame
    overlay = cv2.addWeighted(frame, 1.0, overlay_mask, 1.0, 0)

    # Resize for display
    display_frame = cv2.resize(overlay, displayRez)
    cv2.imshow("Video Frame", display_frame)


    #processing_time = (time.time() - start_time) * 1000  # in milliseconds
    #delay = max(int(1000/fps) - int(processing_time), 1)  # Ensure at least 1 ms delay

    #print(f"Processing time: {processing_time:.2f} ms, Delay: {delay} ms")  # Optional: see timing info

    # make the clock function
    key = cv2.waitKey(0)
    if key == ord('q') & 0xFF: break # If the 'q' key is pressed, exit the loop


#write the last set of filtered points to a CSV file
##!! IMPORTANT !! add your filtered points to .gitignore
with open('filtered_points.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y'])  # Header
    for x, y in filtered_points:
        writer.writerow([x, y])
