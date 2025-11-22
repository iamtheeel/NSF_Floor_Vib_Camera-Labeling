import re
import cv2
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



def evaluate_homography( world_pts, image_pts, H, verbose=True):
    """
    Evaluates homography accuracy by projecting world_pts and comparing to image_pts.

    Args:
        world_pts (np.ndarray): Nx2 array of real-world coordinates.
        image_pts (np.ndarray): Nx2 array of pixel coordinates.
        H (np.ndarray): 3x3 homography matrix.
        verbose (bool): If True, prints per-point and summary errors.

    Returns:
        errors (np.ndarray): Euclidean distance errors between real and projected image points.
        projected_pts (np.ndarray): Nx2 array of projected pixel coordinates.
        world_points (np.ndarray): Nx2 array of real-world coordinates obtained from pixel coordinates.
    """

    image_pts = np.asarray(image_pts, dtype=np.float32)

    # Apply homography to project real-world points to image space
    projected_pts = predict_points_on_image( world_pts, H )

    # Compute Euclidean errors
    errors = np.linalg.norm(image_pts - projected_pts, axis=1)

    if verbose:
        for i, err in enumerate(errors):
            pass
            #print(f"Point {i}: Error = {err:.2f} pixels")


        print(f"\nMean Error: {errors.mean():.2f} pixels")
        print(f"Max Error: {errors.max():.2f} pixels")




    H_inv = np.linalg.inv(H)

    # Extract pixel coordinates from DataFrame
    pixel_pts = image_pts.astype(np.float32)

    # Reshape to (N, 1, 2) for OpenCV
    pixel_pts_reshaped = pixel_pts.reshape(-1, 1, 2)

    # Transform into real-world coordinates (cm)
    world_points = cv2.perspectiveTransform(pixel_pts_reshaped, H_inv).reshape(-1, 2)

    return errors, projected_pts, world_points


def predict_points_on_image(image_pts, H):
    """
    Predicts real-world coordinates from image pixel coordinates using the homography.

    Args:
        image_pts (np.ndarray): Nx2 array of pixel coordinates.
        H (np.ndarray): 3x3 homography matrix.

    Returns:
        world_pts (np.ndarray): Nx2 array of predicted real-world coordinates.
    """
    image_pts = np.asarray(image_pts, dtype=np.float32)
    image_pts_reshaped = image_pts.reshape(-1, 1, 2)

    H_inv = np.linalg.inv(H)
    world_pts = cv2.perspectiveTransform(image_pts_reshaped, H_inv).reshape(-1, 2)
    return world_pts