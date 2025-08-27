####
#   Summer 2025
#   Dr He Lab
###
# Validate Duy's pixel to meter conversion
# Joshua Mehlman
####
from library.camera_calibrator import CameraCalibrator  # Import the calibrator
from library.researcher_base import Researcher          # Duy's config loader

import cv2
import re, os, glob
import csv

researcher = Researcher("josh")
# Init calibrator
calibrator = CameraCalibrator(researcher)
calibrator.load_calibration()

## Initilizations
window_name = "Click to get pixel coords, space to confirm (next image), 'q' to quit"
coords_px = None # Holds the pixel coordinates of the mouse click
# CSV setup (add header if new)
csv_out = f"{researcher.settings.CAMERA_CALIBRATION_DIR}/calibration_validation.csv"
print(f"Writing to CSV: {csv_out}")
csvfile = open(csv_out, "w", newline="") # Make a new file each time
writer = csv.writer(csvfile)
writer.writerow(["file", "gt_x (cm)", "gt_y (cm)", "click_x (px)", "click_y (px)", "calc_x (cm)", "calc_y (cm)"])


## The mouse callback function
def mouse_event(event, x_pix, y_pix, flags, param):
    global coords_px
    if event == cv2.EVENT_MOUSEMOVE:
        # live update crosshair
        display = img.copy()
        cv2.line(display, (x_pix, 0), (x_pix, display.shape[0]), (255, 0, 0), 1)
        cv2.line(display, (0, y_pix), (display.shape[1], y_pix), (255, 0, 0), 1)
        cv2.putText(display, f"({x_pix}, {y_pix})", (x_pix+10, y_pix-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(window_name, display)
    elif event == cv2.EVENT_LBUTTONDOWN:
        coords_px = (x_pix, y_pix)

## The regex pattern to extract ground truth from filename
# Regex to pull x and y in cm"108.3x500y" → (108.3, 500)
pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)y", re.IGNORECASE)
def parse_ground_truth(path):
    name = os.path.basename(path)
    m = pattern.search(name)
    if not m:
        print(f"!!ERROR: No ground truth found in filename: {name}!!")
        return None, None
    x = float(m.group(1))
    y = float(m.group(2))
    return x, y

cv2.namedWindow(window_name)
fileStr = f"{researcher.settings.CAMERA_CALIBRATION_DIR}/*.jpg"
print("Move your mouse to see coordinates. Left click to select .")
print("Press <space> in the image window to confirm location, 'q' to quit.\n")

for imageFile in sorted(glob.glob(fileStr)):
    print(f"Testing image: {imageFile}")

    img = cv2.imread(imageFile)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {imageFile}")
    

    cv2.setMouseCallback(window_name, mouse_event, img) # set the callback function to get the click location

    # Initial display
    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            csvfile.close()
            exit(0)
        elif key == 32:  # spacebar
            gt_cm      = parse_ground_truth(imageFile) # Extract ground truth from filename
            coords_cm  = calibrator.pixel_to_meters(coords_px[0], coords_px[1]) # Get meters from pixels
            coords_cm = (coords_cm[0]*100, coords_cm[1]*100) # convert to cm
            print(f"Image gt: {gt_cm} cm, Clicked at: {coords_px}, From Image: ({coords_cm[0]:.2f}, {coords_cm[1]:.2f})")
            writer.writerow([imageFile, gt_cm[0], gt_cm[1], coords_px[0], coords_px[1], coords_cm[0], coords_cm[1]])
            break

cv2.destroyAllWindows()
csvfile.close()
print(f"\nResults saved to {csv_out}")