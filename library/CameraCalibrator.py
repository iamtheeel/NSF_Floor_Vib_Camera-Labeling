import numpy as np
import cv2 as cv
import glob
import os
import importlib

import pandas as pd

class CameraCalibrator:
    def __init__(self, researcher):
        # Dynamically load the settings module for the given researcher
        self.settings = importlib.import_module(f"settings.{researcher}_settings")

        # Paths
        self.calibration_dir = self.settings.CAMERA_CALIBRATION_DIR
        self.param_file = os.path.join(self.calibration_dir, 'calibration.npz')

        # Checkerboard dimensions: (columns, rows)
        self.checkerboard_dims = (9, 6)

        # Calibration data containers
        self.camMatrix = None
        self.distCoeff = None
        self.rvecs = None
        self.tvecs = None
        self.repError = None
        self.camera_settings = None
        self.homography = None
        self.perspective_transform = None

    def calibrate(self, CAMERA_SETTINGS):
        """
        Performs camera calibration using checkerboard images and saves parameters,
        including the inverse homography matrix assuming Z=0 world plane.
        """
        imgPathList = glob.glob(os.path.join(self.calibration_dir, '*.jpg'))
        print(f'Found {len(imgPathList)} images for calibration.')

        nCols, nRows = self.checkerboard_dims  # OpenCV uses (cols, rows)
        termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 3D world points for checkerboard, assuming Z=0
        worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
        worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

        worldPtsList = []
        imgPtsList = []

        for curImgPath in imgPathList:
            imgBGR = cv.imread(curImgPath)
            imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
            cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

            if cornersFound:
                worldPtsList.append(worldPtsCur)
                cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
                imgPtsList.append(cornersRefined)

        if not worldPtsList or not imgPtsList:
            raise RuntimeError("No valid checkerboard corners found. Calibration failed.")

        # Calibrate camera
        self.repError, self.camMatrix, self.distCoeff, self.rvecs, self.tvecs = cv.calibrateCamera(
            worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
        self.camera_settings = CAMERA_SETTINGS

        # Compute homography for Z=0 using first view
        R, _ = cv.Rodrigues(self.rvecs[0])
        T = self.tvecs[0].reshape(3, 1)
        RT = np.hstack((R[:, :2], T))  # 3x3 matrix: [r1 r2 t]
        H = self.camMatrix @ RT
        self.homography = np.linalg.inv(H)

        # Display results
        print('Camera Matrix:\n', self.camMatrix)
        print('Reprojection Error (pixels): {:.4f}'.format(self.repError))

        # Save calibration results
        np.savez(self.param_file,
                 repError=self.repError,
                 camMatrix=self.camMatrix,
                 distCoeff=self.distCoeff,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs,
                 camera_settings=CAMERA_SETTINGS,
                 homography=self.homography)

    def load_calibration(self):
        """
        Loads camera calibration parameters from the saved .npz file.
        """
        if not os.path.exists(self.param_file):
            raise FileNotFoundError(f"Calibration file not found: {self.param_file}")

        with np.load(self.param_file, allow_pickle=True) as data:
            self.camMatrix = data['camMatrix']
            self.distCoeff = data['distCoeff']
            self.rvecs = data['rvecs']
            self.tvecs = data['tvecs']
            self.repError = data['repError']
            self.camera_settings = data['camera_settings'].item()
            self.homography = data.get('homography', None)

        print("Calibration parameters loaded.")

    def pixel_to_meters(self, x_pixel, y_pixel, perspective=False):
        """
        Converts a pixel location (x_pixel, y_pixel) to real-world coordinates (X, Y) in meters
        assuming the point lies on the Z=0 plane.

        If perspective=True, uses the perspective transform (from CSV correspondences).
        Otherwise, uses the calibration-based inverse homography.
        """
        if perspective:
            # Ensure perspective transform is available
            if self.perspective_transform is None:
                self.load_calibration()  # loads perspective_transform if saved
            assert self.perspective_transform is not None, "Perspective transform not available."

            transform = self.perspective_transform
        else:
            if self.homography is None:
                self.load_calibration()
            assert self.homography is not None, "Homography could not be loaded."
            transform = self.homography

        assert transform.shape == (3, 3), "Transform matrix must be 3x3."

        pixel_coords = np.array([float(x_pixel), float(y_pixel), 1.0])
        world_coords = transform @ pixel_coords
        world_coords /= world_coords[2]  # Normalize homogeneous coords

        x_meters, y_meters = world_coords[0], world_coords[1]
        return x_meters, y_meters


    def calculate_perspective_transform(self):
        """
        Computes a perspective transform matrix from pixel to real-world coordinates
        using manually defined correspondences in perspectiveTransform.csv.
        Saves the result in memory and to the calibration file.
        """
        if self.perspective_transform is not None:
            print("Perspective transform already calculated.")
            return

        # Load existing calibration if needed (to get param_file path)
        if self.camMatrix is None:
            self.load_calibration()

        csv_path = os.path.join(self.calibration_dir, "perspectiveTransform.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Check required columns
        # u_pixel is the horizontal pixel coordinate
        # v_pixel is the vertical pixel coordinate
        required_cols = {'u_pixel', 'v_pixel', 'x_meters', 'y_meters'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        src_pts = df[['u_pixel', 'v_pixel']].values.astype(np.float32)
        dst_pts = df[['x_meters', 'y_meters']].values.astype(np.float32)

        if len(src_pts) < 4:
            raise ValueError("At least 4 point correspondences are required for homography.")

        # Compute perspective transform (homography)
        H, status = cv.findHomography(src_pts, dst_pts)
        self.perspective_transform = H

        # Save perspective transform to calibration file
        with np.load(self.param_file, allow_pickle=True) as data:
            save_dict = dict(data)
            save_dict['perspective_transform'] = self.perspective_transform

        np.savez(self.param_file, **save_dict)
        print("Perspective transform calculated and saved.")
        

    def remove_distortion(self, start_time=0, end_time=None, output_filename="undistorted_output.avi"):
        """
        Removes lens distortion from a video file using the camera calibration parameters.
        """
        if self.camMatrix is None or self.distCoeff is None:
            self.load_calibration()

        video_path = os.path.join(self.settings.FOLDER_DIR, self.settings.VIDEO_FILENAME)
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames

        output_path = os.path.join(self.calibration_dir, output_filename)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        camMatrixNew, _ = cv.getOptimalNewCameraMatrix(self.camMatrix, self.distCoeff, (width, height), 1, (width, height))

        print(f"Processing video from frame {start_frame} to {end_frame}...")

        while cap.isOpened():
            frame_id = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            if frame_id >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            undistorted_frame = cv.undistort(frame, self.camMatrix, self.distCoeff, None, camMatrixNew)
            out.write(undistorted_frame)

        cap.release()
        out.release()
        print(f"Undistorted video saved to {output_path}")





