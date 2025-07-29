import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import importlib

class CameraCalibrator:
    def __init__(self, researcher):
        self.settings = importlib.import_module(f"settings.{researcher}_settings")

        self.calibration_dir = self.settings.CAMERA_CALIBRATION_DIR
        self.param_file = os.path.join(self.calibration_dir, 'calibration.npz')
        
        self.checkerboard_dims = (9, 6)
        self.camMatrix = None
        self.distCoeff = None
        self.rvecs = None
        self.tvecs = None
        self.repError = None
        self.camera_settings = None

    def calibrate(self, CAMERA_SETTINGS):
        imgPathList = glob.glob(os.path.join(self.calibration_dir, '*.jpg'))
        print(f'Found {len(imgPathList)} images for calibration.')

        nRows, nCols = self.checkerboard_dims
        termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
        worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

        worldPtsList = []
        imgPtsList = []

        for curImgPath in imgPathList:
            imgBGR = cv.imread(curImgPath)
            imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
            cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

            if cornersFound:
                worldPtsList.append(worldPtsCur)
                cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
                imgPtsList.append(cornersRefined)

        self.repError, self.camMatrix, self.distCoeff, self.rvecs, self.tvecs = cv.calibrateCamera(
            worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
        self.camera_settings = CAMERA_SETTINGS

        print('Camera Matrix: \n', self.camMatrix)
        print('Reproj Error (pixels): {:.4f}'.format(self.repError))

        np.savez(self.param_file,
                 repError=self.repError,
                 camMatrix=self.camMatrix,
                 distCoeff=self.distCoeff,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs,
                 camera_settings=CAMERA_SETTINGS)

    def load_calibration(self):
        with np.load(self.param_file) as data:
            self.camMatrix = data['camMatrix']
            self.distCoeff = data['distCoeff']
            self.rvecs = data['rvecs']
            self.tvecs = data['tvecs']
            self.repError = data['repError']
        print("Calibration parameters loaded.")

    def remove_distortion(self, start_time=0, end_time=None, output_filename="undistorted_output.avi"):
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





