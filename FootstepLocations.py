from cv2Utils import *

# Constants for file paths
FOLDER_DIR = r"E:\Documents2TBm2\StatisticDocuments\GaitResearch\Box\StudentData\25_07_10"
VIDEO_FILENAME = r"intercept_run_7-10-2025_10-45-46 AM.asf"
HDF5_FILENAME = r"Jack_clockTest_interuptVPoll.hdf5"

VIDEO_DIR = os.path.join(FOLDER_DIR, VIDEO_FILENAME)
HDF5_DIR = os.path.join(FOLDER_DIR, HDF5_FILENAME)

foot_locations = get_foot_data(VIDEO_DIR)
footstep_times = get_footstep_times(HDF5_DIR)
footstep_y = append_avg_foot_positions(footstep_times, foot_locations)
footstep_y.to_csv("footstep_times_with_avg.csv", index=False)