import re
import cv2
import os
import pandas as pd
import numpy as np




# Not Used anymore
def calculate_homography_from_checkerboard(image_path, checkerboard_size=(6, 9), square_size=15):
    """
    Calculates the homography from real-world checkerboard points to image points.

    Args:
        image_path (str): Path to the checkerboard image.
        checkerboard_size (tuple): (cols, rows) number of inner corners.
        square_size (float): Size of each square in real-world units (mm, cm, etc.)

    Returns:
        H (np.ndarray): 3x3 Homography matrix (real-world → image)
        image (np.ndarray): Original BGR image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size)
    if not ret:
        raise RuntimeError("Checkerboard not detected")

    # Refine corners to subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Real-world points in 2D plane
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # scale to physical units

    # Compute homography: world → image
    image_points = corners.reshape(-1, 2)
    world_points = objp[:, :2]
    H, _ = cv2.findHomography(world_points, image_points)

    return H



def find_checkerboard_corners_all_images(image_dir, pattern_size=(9, 6), square_size=15):
    """
    Reads images from a folder and detects checkerboard corners.

    Args:
        image_dir (str): Path to folder containing images.
        pattern_size (tuple): Number of inner corners (cols, rows).

    Returns:
        pd.DataFrame: DataFrame with columns: [file, corner_index, x, y]
    """
    all_points = []
    filenames = sorted(os.listdir(image_dir))

    print(f"[INFO] Found {len(filenames)} files in {image_dir}")
    print(filenames)
    for filename in filenames:
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        filepath = os.path.join(image_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"[WARN] Could not read image: {filepath}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        
        # remove file extension
        base_name = os.path.splitext(filename)[0]

        # Find all numbers (including decimals)
        numbers = re.findall(r"[\d.]+", base_name)  # Exclude the file extension

        # Convert them to floats or ints as appropriate
        real_x, real_y  = [float(n) for n in numbers]


        #real_x = float( filename.split('x')[0] )
        #real_y = float( filename.split('x')[1].split('y')[0])

        if found:
            corners = corners.squeeze(axis=1)  # shape (N, 2)

            

            for idx, (x, y) in enumerate(corners):

                current_column = idx % pattern_size[0]
                current_row =  pattern_size[1] - (idx // pattern_size[0] ) - 1

                all_points.append({
                    'file': filename,
                    'corner_index': idx, 
                    'x': float(x),
                    'y': float(y),
                    'real_x': real_x + square_size + current_column * square_size,  # Adjust for inner corner
                    'real_y': real_y + square_size + (current_row * square_size)   # Adjust for inner corner
                })
            print(f"[INFO] Added Successfully: {filename}")
        else:
            print(f"[INFO] Checkerboard not found in: {filename}")

        

    df = pd.DataFrame(all_points)
    return df


def compute_homography_from_dataframe(df):
    """
    Computes the homography from real-world coordinates to image coordinates.

    Args:
        df (pd.DataFrame): Must contain columns 'x', 'y', 'real_x', 'real_y'

    Returns:
        H (np.ndarray): 3x3 homography matrix mapping (real_x, real_y) → (x, y)
    """
    required_cols = {'x', 'y', 'real_x', 'real_y'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    image_points = df[['x', 'y']].values.astype(np.float32)
    world_points = df[['real_x', 'real_y']].values.astype(np.float32)

    if len(image_points) < 4:
        raise ValueError("At least 4 point pairs are required to compute a homography.")

    H, status = cv2.findHomography(world_points, image_points, method=0)
    return H




