# crop_logic.py
# Utilities for cropping and smoothing frame regions based on pose landmarks

import numpy as np

# === Example Crop Logic ===

def crop_to_square(frame, landmarks, direction, maintain_dim):
    """
    Calculates crop dimensions based on landmarks.
    Returns updated min_w, max_w, min_h, max_h, maintain_dim.
    """
    h, w, _ = frame.shape

    # Example center: between hips (landmarks 23, 24)
    center_x = int((landmarks[23].x + landmarks[24].x) / 2 * w)
    center_y = int((landmarks[23].y + landmarks[24].y) / 2 * h)

    # Crop square around this center
    size = min(h, w) // 2

    min_w = max(center_x - size, 0)
    max_w = min(center_x + size, w)
    min_h = max(center_y - size, 0)
    max_h = min(center_y + size, h)

    maintain_dim = [min_h, max_h, min_w, max_w]
    return min_w, max_w, min_h, max_h, maintain_dim


def smooth_crop_dim(prev_dims, min_w, max_w, min_h, max_h, alpha=0.1):
    """
    Applies exponential smoothing to crop dimensions to avoid jitter.
    prev_dims: [prev_min_h, prev_max_h, prev_min_w, prev_max_w]
    """
    if prev_dims[0] is None:
        return [min_h, max_h, min_w, max_w], min_w, max_w, min_h, max_h

    sm_min_h = int(prev_dims[0] * (1 - alpha) + min_h * alpha)
    sm_max_h = int(prev_dims[1] * (1 - alpha) + max_h * alpha)
    sm_min_w = int(prev_dims[2] * (1 - alpha) + min_w * alpha)
    sm_max_w = int(prev_dims[3] * (1 - alpha) + max_w * alpha)

    return [sm_min_h, sm_max_h, sm_min_w, sm_max_w], sm_min_w, sm_max_w, sm_min_h, sm_max_h


def crop_to_Southhall():
    # Placeholder hall logic
    return 0, 1920, 0, 1080, "South"


def crop_to_Northhall():
    # Placeholder hall logic
    return 0, 1920, 0, 1080, "North"


def landmarks_of_fullscreen(landmarks, min_w, max_w, min_h, max_h):
    """Placeholder function for adjusting fullscreen landmark alignment."""
    return
