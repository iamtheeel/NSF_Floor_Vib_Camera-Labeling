####
#   Summer 2025
#   Dr He Lab
###
# Validate Duy's pixel to meter conversion
# From a csv output by testXY.py
# Joshua Mehlman
####

from library.camera_calibrator import CameraCalibrator  # Import the calibrator
from library.researcher_base import Researcher          # Duy's config loader

import csv

researcher = Researcher("duy")
# Init calibrator
calibrator = CameraCalibrator(researcher)
calibrator.load_calibration()


csv_in = f"{researcher.settings.CAMERA_CALIBRATION_DIR}/calibration_validation_1.csv"
csv_out = f"{researcher.settings.CAMERA_CALIBRATION_DIR}/calibration_validation.csv"


with open(csv_in, newline='') as input_csvfile, open(csv_out, 'w', newline='') as output_csvfile:
    reader = csv.DictReader(input_csvfile)
    fieldnames = reader.fieldnames # Get headder from input file

    writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Step through line by line
    for row in reader:
        # Get pixel coords from CSV
        x_px = int(row["click_x (px)"])
        y_px = int(row["click_y (px)"])

        # Calculate meters from pixels
        coords_cm  = calibrator.mixed_pixel_to_meters(x_pixel=x_px, y_pixel=y_px) # Get meters from pixels

        # Update the row with calculated values
        row["calc_x (cm)"] = coords_cm[0] * 100  # Convert to cm
        row["calc_y (cm)"] = coords_cm[1] * 100  # Convert to cm 
        writer.writerow(row)