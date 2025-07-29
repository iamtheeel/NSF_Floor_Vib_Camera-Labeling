import importlib
from library.CameraCalibrator import CameraCalibrator

# load researcher settings found in settings directory
RESEARCHER = "duy"
cc = CameraCalibrator(RESEARCHER)

# select script to run
RUN_CALIBRATION = False
RUN_DISTORTION_REMOVAL = True

"""
To calibrate new camera settings,
Create new folder in cameraCalibration directory,
update RESEARCHER_settings.py with new CAMERA_CALIBRATION_DIR settings,
drop all checkerboard images in the new folder,
update CAMERA_SETTINGS with new values,
and run this script.
"""
if RUN_CALIBRATION:
    # settings to be saved in calibration file
    CAMERA_SETTINGS = {
    "pan": -9.45,
    "tilt": -19.50,
    "zoom": 1,
    "focus": 875,
    "autofocus": "on"
}
    cc.calibrate(CAMERA_SETTINGS)

# undistort the video specified in RESEARCHER_settings.py
if RUN_DISTORTION_REMOVAL:
    cc.remove_distortion(start_time=15, end_time=20)