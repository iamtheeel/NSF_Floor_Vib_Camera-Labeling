import numpy as np
import cv2 as cv
import glob
import os
import importlib
import pandas as pd
from library.researcher_base import Researcher

class CameraCalibrator:
    def __init__(self, researcher: Researcher):
        # Initialize with a Researcher instance
        self.researcher = researcher
        self.settings = researcher.settings

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
        self.perspective_transform_undistorted = None


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
            self.perspective_transform = data.get('perspective_transform', None)
            self.perspective_transform_undistorted = data.get('perspective_transform_undistorted', None)

        print("Calibration parameters loaded.")


    def pixel_to_meters(self, x_pixel, y_pixel, perspective=True):
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

    def pixel_to_meters_undistorted(self, x_pixel, y_pixel):
        """
        Converts a pixel location from an undistorted image to real-world coordinates (X, Y) in meters.
        Uses the undistorted perspective transform if available; otherwise, uses the original homography.

        Args:
            x_pixel, y_pixel: Pixel coordinates in undistorted image.

        Returns:
            (x_meters, y_meters): Real-world coordinates on Z=0 plane.
        """
        if self.camMatrix is None or self.distCoeff is None:
            self.load_calibration()

        # Step 1: Undistort the input pixel point
        pixel = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        undistorted = cv.undistortPoints(pixel, self.camMatrix, self.distCoeff, P=self.camMatrix)
        x_u, y_u = undistorted[0, 0]

        # Step 2: Choose the transform
        if self.perspective_transform_undistorted is None:
            self.load_calibration()

        if self.perspective_transform_undistorted is not None:
            transform = self.perspective_transform_undistorted
        elif self.homography is not None:
            transform = self.homography
        else:
            raise RuntimeError("No suitable transform available (neither undistorted nor original homography).")

        # Step 3: Map undistorted pixel to world coordinates
        pixel_coords = np.array([x_u, y_u, 1.0])
        world_coords = transform @ pixel_coords
        world_coords /= world_coords[2]

        return world_coords[0], world_coords[1]



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

        vid = self.researcher.load_video()
        if vid is None:
            return

        meta = self.researcher.video_metadata
        fps = meta["fps"]
        width = meta["width"]
        height = meta["height"]
        total_frames = meta["total_frames"]

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames

        output_path = os.path.join(self.calibration_dir, output_filename)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        vid.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        camMatrixNew, _ = cv.getOptimalNewCameraMatrix(self.camMatrix, self.distCoeff, (width, height), 1, (width, height))

        print(f"Processing video from frame {start_frame} to {end_frame}...")

        while vid.isOpened():
            frame_id = int(vid.get(cv.CAP_PROP_POS_FRAMES))
            if frame_id >= end_frame:
                break

            ret, frame = vid.read()
            if not ret:
                break

            undistorted_frame = cv.undistort(frame, self.camMatrix, self.distCoeff, None, camMatrixNew)
            out.write(undistorted_frame)

        vid.release()
        out.release()
        print(f"Undistorted video saved to {output_path}")

    def undistort_folder(self, input_folder, _=None):  # output_folder argument no longer needed
        """
        Undistorts all images in a folder using the camera calibration parameters.
        Saves the undistorted images to a subfolder named 'undistorted' inside the input folder.
        """
        if self.camMatrix is None or self.distCoeff is None:
            self.load_calibration()

        # Define the undistorted subfolder path
        output_folder = os.path.join(input_folder, 'undistorted')
        os.makedirs(output_folder, exist_ok=True)

        image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + glob.glob(os.path.join(input_folder, '*.png'))

        for img_path in image_files:
            img = cv.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            undistorted_img = cv.undistort(img, self.camMatrix, self.distCoeff)
            output_path = os.path.join(output_folder, os.path.basename(img_path))
            cv.imwrite(output_path, undistorted_img)
            print(f"Undistorted image saved: {output_path}")

    def draw_meter_grid(self, image, x_range, y_range, step=1.0, perspective=False):
        """
        Draws '+' markers on the image in a real-world meter grid.

        Args:
            image: BGR image (np.ndarray) to draw on.
            x_range: (x_min, x_max) in meters
            y_range: (y_min, y_max) in meters
            step: grid spacing (default: 1 meter)
            perspective: whether to use perspective transform (True) or calibration homography (False)
        Returns:
            Image with grid drawn.
        """
        if perspective:
            if self.perspective_transform is None:
                self.load_calibration()
            H = self.perspective_transform
        else:
            if self.homography is None:
                self.load_calibration()
            H = self.homography

        assert H is not None, "No transformation matrix available."
        H_inv = np.linalg.inv(H)  # To convert real-world (meters) → pixel

        output = image.copy()
        color = (0, 0, 255)  # red
        thickness = 10
        marker_size = 75

        x_min, x_max = x_range
        y_min, y_max = y_range

        for x in np.arange(x_min, x_max + step, step):
            for y in np.arange(y_min, y_max + step, step):
                world_pt = np.array([x, y, 1.0])
                pixel_pt = H_inv @ world_pt
                pixel_pt /= pixel_pt[2]

                u, v = int(round(pixel_pt[0])), int(round(pixel_pt[1]))

                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv.drawMarker(output, (u, v), color, markerType=cv.MARKER_CROSS,
                                markerSize=marker_size, thickness=thickness)

        return output

    def draw_meter_grid_undistorted(self, image, x_range, y_range, step=1.0):
        """
        Draws '+' markers on the undistorted image using real-world meter grid.

        If an undistorted perspective transform is available, it is used to map
        world coordinates (X, Y) → undistorted pixel coordinates (u, v).
        Otherwise, cv.projectPoints() is used.

        Args:
            image: Undistorted BGR image (np.ndarray) to draw on.
            x_range: (x_min, x_max) in meters
            y_range: (y_min, y_max) in meters
            step: Grid spacing in meters

        Returns:
            Image with meter grid drawn.
        """
        if self.camMatrix is None or self.distCoeff is None:
            self.load_calibration()

        if self.perspective_transform_undistorted is None:
            self.load_calibration()

        output = image.copy()
        color = (0, 0, 255)  # Red
        thickness = 10
        marker_size = 75

        x_min, x_max = x_range
        y_min, y_max = y_range

        # Generate 2D world points (Z=0 plane)
        world_pts = []
        for x in np.arange(x_min, x_max + step, step):
            for y in np.arange(y_min, y_max + step, step):
                world_pts.append([x, y])
        world_pts = np.array(world_pts, dtype=np.float32)

        if self.perspective_transform_undistorted is not None:
            # Use the undistorted perspective transform (world → pixel)
            H = self.perspective_transform_undistorted
            for pt in world_pts:
                world_h = np.array([pt[0], pt[1], 1.0])
                pixel_h = H @ world_h
                pixel_h /= pixel_h[2]
                u, v = int(round(pixel_h[0])), int(round(pixel_h[1]))
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv.drawMarker(output, (u, v), color, markerType=cv.MARKER_CROSS,
                                markerSize=marker_size, thickness=thickness)
        else:
            # Fall back to projection using extrinsics
            if self.rvecs is None or self.tvecs is None:
                self.load_calibration()

            world_pts_3d = np.hstack([world_pts, np.zeros((world_pts.shape[0], 1))])
            img_pts, _ = cv.projectPoints(
                world_pts_3d,
                self.rvecs[0],
                self.tvecs[0],
                self.camMatrix,
                np.zeros_like(self.distCoeff)  # Image is already undistorted
            )
            for pt in img_pts:
                u, v = map(int, map(round, pt.ravel()))
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv.drawMarker(output, (u, v), color, markerType=cv.MARKER_CROSS,
                                markerSize=marker_size, thickness=thickness)

        return output



    def piecewise_pixel_to_meters(self, x_pixel, y_pixel, threshold_y=500, transition_width=30):
        """
        Converts pixel to real-world meters using a piecewise-smooth blend between
        calibration-based and perspective-based transforms.

        Uses:
        - Perspective transform for pixels clearly above the threshold.
        - Calibration transform for pixels clearly below the threshold.
        - Linear interpolation in the transition band to avoid discontinuity.

        Args:
            x_pixel, y_pixel: pixel coordinates in image
            threshold_y: center pixel row for switching transforms (e.g., 500)
            transition_width: half-width of the blend zone around threshold_y

        Returns:
            (x_meters, y_meters): real-world coordinates in meters
        """
        # Ensure both transforms are available
        if self.homography is None or self.perspective_transform is None:
            self.load_calibration()

        assert self.homography is not None, "Homography not loaded."
        assert self.perspective_transform is not None, "Perspective transform not loaded."

        # Prepare pixel coordinates (homogeneous)
        pixel_coords = np.array([float(x_pixel), float(y_pixel), 1.0])

        # Apply both transforms
        world_p = self.perspective_transform @ pixel_coords
        world_p /= world_p[2]
        x_p, y_p = world_p[0], world_p[1]

        world_h = self.homography @ pixel_coords
        world_h /= world_h[2]
        x_h, y_h = world_h[0], world_h[1]

        # Compute alpha for blending
        if y_pixel <= threshold_y - transition_width:
            return x_p, y_p
        elif y_pixel >= threshold_y + transition_width:
            return x_h, y_h
        else:
            # Linear blend in the transition zone
            alpha = (y_pixel - (threshold_y - transition_width)) / (2 * transition_width)
            x = (1 - alpha) * x_p + alpha * x_h
            y = (1 - alpha) * y_p + alpha * y_h
            return x, y

    def calculate_perspective_transform_undistorted(self):
        """
        Computes a perspective transform for undistorted images using the original
        perspectiveTransform.csv correspondences. Undistorts pixel points before computing.
        Saves the matrix to self.perspective_transform_undistorted and updates calibration file.
        """
        if self.camMatrix is None or self.distCoeff is None:
            self.load_calibration()

        csv_path = os.path.join(self.calibration_dir, "perspectiveTransform.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {'u_pixel', 'v_pixel', 'x_meters', 'y_meters'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        pixel_pts = df[['u_pixel', 'v_pixel']].astype(np.float32).values
        world_pts = df[['x_meters', 'y_meters']].astype(np.float32).values

        if len(pixel_pts) < 4:
            raise ValueError("At least 4 point correspondences are required for homography.")

        # Undistort pixel coordinates
        pixel_pts_undistort = cv.undistortPoints(
            pixel_pts.reshape(-1, 1, 2), self.camMatrix, self.distCoeff, P=self.camMatrix
        ).reshape(-1, 2)

        # Compute homography: world → undistorted image
        H_undistorted, _ = cv.findHomography(world_pts, pixel_pts_undistort)

        if H_undistorted is None:
            raise RuntimeError("Failed to compute undistorted homography.")

        self.perspective_transform_undistorted = H_undistorted

        # Save into calibration file
        with np.load(self.param_file, allow_pickle=True) as data:
            save_dict = dict(data)
            save_dict['perspective_transform_undistorted'] = H_undistorted

        np.savez(self.param_file, **save_dict)
        print("Undistorted perspective transform calculated and saved.")





