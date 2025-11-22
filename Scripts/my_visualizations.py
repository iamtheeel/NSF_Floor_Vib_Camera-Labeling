import re
import cv2
import os
import pandas as pd
import numpy as np


import csv
import imageio

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score # type: ignore



def draw_points_with_indices(image_path, df, radius=5, font_scale=0.5, color=(0, 255, 0)):
    """
    Draws detected checkerboard points on an image, labeled with index and real-world position.

    Args:
        image_path (str): Path to the image file.
        df (pd.DataFrame): DataFrame containing columns ['x', 'y', 'corner_index', 'real_x', 'real_y'].
        radius (int): Circle radius for each point.
        font_scale (float): Font size for index labels.
        color (tuple): BGR color for points and text.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for _, row in df.iterrows():
        x, y = int(row['x']), int(row['y'])
        real_x, real_y = row['real_x'], row['real_y']
        label = f"{int(row['corner_index'])}: ({real_x:.1f}, {real_y:.1f})"

        

        if _ % 2 == 1:  # To avoid clutter, only label every second point with real-world coords
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.putText(img, f"{int(row['corner_index'])}", (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"({real_x:.1f}, {real_y:.1f})", (x-50, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100,100,100), 1, cv2.LINE_AA)

    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    


def draw_real_world_grid_on_image(image, H):
    """
    Draws a perspective-correct real-world grid with 25 cm minor spacing and 100 cm major spacing.
    Adds semi-transparent overlay blending so it looks like part of the scene.
    """
    x_range = (0, 315)
    y_range = (0, 3000)
    minor_spacing = 25                   # 25 cm minor grid
    major_spacing = 100                  # 100 cm major grid

    # colors tuned for ground overlay
    color_minor=(120, 120, 255) 
    color_major=(60, 60, 255)
    alpha = 0.4                       # transparency strength

    thickness_minor = 1
    thickness_major = 2
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1

    # --- prepare overlay (same size as image) ---
    overlay = image.copy()

    # Generate coordinate arrays (inclusive)
    x_minor = np.arange(x_range[0], x_range[1] + minor_spacing, minor_spacing, dtype=float)
    y_minor = np.arange(y_range[0], y_range[1] + minor_spacing, minor_spacing, dtype=float)
    x_major = np.arange(x_range[0], x_range[1] + major_spacing, major_spacing, dtype=float)
    y_major = np.arange(y_range[0], y_range[1] + major_spacing, major_spacing, dtype=float)

    def draw_lines(img_target, x_vals, y_vals, color, thickness):
        # verticals
        for x in x_vals:
            world_pts = np.float32([[x, y] for y in y_vals]).reshape(-1, 1, 2) # type: ignore
            img_pts = cv2.perspectiveTransform(world_pts, H).reshape(-1, 2)
            for i in range(len(img_pts) - 1):
                cv2.line(img_target,
                         tuple(np.int32(img_pts[i])), # type: ignore
                         tuple(np.int32(img_pts[i + 1])), # type: ignore
                         color, thickness, cv2.LINE_AA)
        # horizontals
        for y in y_vals:
            world_pts = np.float32([[x, y] for x in x_vals]).reshape(-1, 1, 2) # type: ignore
            img_pts = cv2.perspectiveTransform(world_pts, H).reshape(-1, 2)
            for i in range(len(img_pts) - 1):
                cv2.line(img_target,
                         tuple(np.int32(img_pts[i])), # type: ignore
                         tuple(np.int32(img_pts[i + 1])), # type: ignore
                         color, thickness, cv2.LINE_AA)

    # --- draw both grids onto overlay ---
    draw_lines(overlay, x_minor, y_minor, color_minor, thickness_minor)
    draw_lines(overlay, x_major, y_major, color_major, thickness_major)

    # --- labels along bottom and left ---
    label_points = []
    for x in x_major:
        label_points.append([x, y_range[0]])
    for y in y_major[1:]:
        label_points.append([x_range[0], y])

    label_points = np.float32(label_points).reshape(-1, 1, 2) # type: ignore
    image_points = cv2.perspectiveTransform(label_points, H).reshape(-1, 2)

    for (wx, wy), (ix, iy) in zip(label_points.reshape(-1, 2), image_points):
        label = f"{int(wy)}"
        cv2.putText(
            overlay,
            label,
            (int(ix) - 50, int(iy) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # --- blend overlay with transparency ---
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return blended




def draw_graph_error( gt, pred ):


    #gt = test_data[['gt_x (cm)', 'gt_y (cm)']].values
    #pred = pred_cm

    # Extract coordinates
    # Extract coordinates
    gt_x, gt_y = gt[:, 0], gt[:, 1]
    pred_x, pred_y = pred[:, 0], pred[:, 1]

    # --- Linear fit (Y = aX + b) ---
    fit_x = np.polyfit(gt_x, pred_x, 1)
    fit_y = np.polyfit(gt_y, pred_y, 1)
    a_x, b_x = fit_x
    a_y, b_y = fit_y

    # --- Generate fitted lines ---
    fit_line_x = np.polyval(fit_x, gt_x)
    fit_line_y = np.polyval(fit_y, gt_y)

    # --- R² values ---
    r2_x = r2_score(gt_x, pred_x)
    r2_y = r2_score(gt_y, pred_y)

    # --- Plot style ---
    plt.figure(figsize=(12, 5))

    # =====================================
    # Distance Down Hall (Y direction)
    # =====================================
    plt.subplot(1, 2, 1)
    plt.scatter(gt_y, pred_y, color='blue', label='Data')
    plt.plot(gt_y, fit_line_y, color='black', linewidth=1.5, linestyle='--', label='Fit')

    plt.xlabel("Read From Image Label (cm)")
    plt.ylabel("Calculated From Image (cm)")
    plt.title("Distance Down Hall\nGenerated from: calibrator.pixel_to_meters")

    # Equation + R² box
    text = f"f(x) = {a_y:.4f}x + {b_y:.4f}\n$R^2$ = {r2_y:.4f}"
    plt.text(0.05, 0.90, text,
            transform=plt.gca().transAxes,
            fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))
    plt.grid(True)
    plt.legend()

    # =====================================
    # Distance Across Hall (X direction)
    # =====================================
    plt.subplot(1, 2, 2)
    plt.scatter(gt_x, pred_x, color='orangered', label='Data')
    plt.plot(gt_x, fit_line_x, color='black', linewidth=1.5, linestyle='--', label='Fit')

    plt.xlabel("Read From Image Label (cm)")
    plt.ylabel("Calculated From Image (cm)")
    plt.title("Distance Across Hall\nGenerated from: calibrator.pixel_to_meters")

    # Equation + R² box
    text = f"f(x) = {a_x:.4f}x + {b_x:.4f}\n$R^2$ = {r2_x:.4f}"
    plt.text(0.05, 0.90, text,
            transform=plt.gca().transAxes,
            fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



def draw_points_on_image(
        

    image,
    points,
    color=(199, 0, 0),
    radius=5,
    thickness=-1,
    labels=None,
    label_color=(0, 255, 0),
    label_offset=(10, -10),
    copy=True
):
    """
    Draws points on an image using OpenCV.

    Args:
        image (str | np.ndarray): Path to image or already-loaded image (BGR or RGB).
        points (array-like): Nx2 array/list of pixel coordinates [(x, y), ...].
        color (tuple): BGR color for points (default red).
        radius (int): Radius of the circle.
        thickness (int): Thickness of circle (-1 for filled).
        labels (list[str], optional): Labels for each point (same length as points).
        label_color (tuple): BGR color for text labels.
        label_offset (tuple): Offset (dx, dy) for label text relative to point.
        copy (bool): If True, returns a copy instead of modifying in place.

    Returns:
        np.ndarray: Image with points drawn.
    """
    # Load if given a file path
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    else:
        img = image.copy() if copy else image

    # Ensure numpy array
    pts = np.array(points, dtype=int)

    # Draw points
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), radius, color, thickness)

        if labels is not None and i < len(labels):
            label = str(labels[i])
            cv2.putText(
                img, label,
                (x + label_offset[0], y + label_offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, label_color, 2, cv2.LINE_AA
            )

    

    return img




if __name__ == "__main__":

    import my_visualizations as myviz
    import camera_calibration as setcal
    import predictions as pred


    image_dir = 'Calibration_Images'
    csv_calibration_file = r'csv_files\all_checkerboard_points.csv'

    #setcal.find_checkerboard_corners_all_images(image_dir).to_csv(csv_calibration_file, index=False)
    df = pd.read_csv(csv_calibration_file)



    H = setcal.compute_homography_from_dataframe( df )


    # ------ Load Test Data ------
    csv_hand_validation_file = r'csv_files\hand_calculated.csv'
    test_data = pd.read_csv(csv_hand_validation_file)
    # filter test data so only has x values less than 100
    # This is because some of the hand calculated points are measured incorrectly
    test_data = test_data[ test_data['gt_x (cm)'] < 100 ]
    #--------------------------------

    

    # ------ Visualization and Evaluation ------

    test_image = "0x900y.jpg"
    all_points_pixel = np.append( df[['x', 'y']].values, test_data[['click_x (px)', 'click_y (px)']].values, axis=0 )
    all_points_real = np.append( df[['real_x', 'real_y']].values, test_data[['gt_x (cm)', 'gt_y (cm)']].values, axis=0 )
    
    img = myviz.draw_points_on_image(f'{image_dir}\\{test_image}', all_points_pixel)
    img = myviz.draw_real_world_grid_on_image(
            img, H
        )
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'All Points and Grid Overlay {test_image}')
    plt.show()
    
    error, pred_pixels, pred_cm = pred.evaluate_homography( test_data[['gt_x (cm)', 'gt_y (cm)']].values, test_data[['click_x (px)', 'click_y (px)']].values, H)
    
    myviz.draw_graph_error(
        test_data[['gt_x (cm)', 'gt_y (cm)']].values,
        pred_cm )
    
    error, pred_pixels, pred_cm = pred.evaluate_homography( all_points_real, all_points_pixel, H)
    
    myviz.draw_graph_error(
        all_points_real,
        pred_cm )

    # --------------------------------------------

# End