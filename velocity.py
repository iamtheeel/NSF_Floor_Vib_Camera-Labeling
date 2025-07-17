from collections import defaultdict
import numpy as np

# === Velocity: Left/Right Heels and Toes ===

def compute_velocity_and_r2(label, data):
    times = np.array([t for t, d in data])
    dists = np.array([d for t, d in data])
    if len(times) < 2:
        return (label, None, None)

    A = np.vstack([times, np.ones_like(times)]).T #design matrix A for linear regression using least squares
    m, b = np.linalg.lstsq(A, dists, rcond=None)[0] #linear least squares function
    y_pred = m * times + b
    ss_res = np.sum((dists - y_pred) ** 2) #residual sum of squares
    ss_tot = np.sum((dists - np.mean(dists)) ** 2) #total sum of squares
    r2 = 1 - (ss_res / ss_tot) #1.0 = perfect fit, 0.0 = no fit

    print(f"✅ {label}: velocity = {m:.4f} m/s | R² = {r2:.4f}")
    return (label, m, r2)

# === Velocity: Average Heels ===
def calculate_avg_heel_velocity(landmark_data, verbose=True): #verbose controls whether the function prints progress or result messages to the console.
                                                              #If true, it will print messages; if false, it will not.
    left = landmark_data.get("LeftHeel", [])
    right = landmark_data.get("RightHeel", [])

    # Align timestamps
    times_left = {t for t, _ in left}
    times_right = {t for t, _ in right}
    common_times = sorted(times_left & times_right) #only compare Left and Right distances from the same video frames.

    if len(common_times) < 2:
        if verbose:
            print("⚠️ AvgHeels: Not enough common timestamps.")
        return ("AvgHeels", None, None)

    left_dict = dict(left)
    right_dict = dict(right)

    avg_distances = [(t, (left_dict[t] + right_dict[t]) / 2) for t in common_times]
    times = np.array([t for t, _ in avg_distances])
    dists = np.array([d for _, d in avg_distances])

    A = np.vstack([times, np.ones_like(times)]).T
    m, b = np.linalg.lstsq(A, dists, rcond=None)[0]
    y_pred = m * times + b
    ss_res = np.sum((dists - y_pred) ** 2)
    ss_tot = np.sum((dists - np.mean(dists)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    if verbose:
        print(f"✅ AvgHeels: velocity = {m:.4f} m/s | R² = {r2:.4f}")

    return ("AvgHeels", m, r2)

# === Velocity: Average Toes ===
def calculate_avg_toe_velocity(landmark_data, verbose=True):
    left = landmark_data.get("LeftToe", [])
    right = landmark_data.get("RightToe", [])

    # Align timestamps
    times_left = {t for t, _ in left}
    times_right = {t for t, _ in right}
    common_times = sorted(times_left & times_right)

    if len(common_times) < 2:
        if verbose:
            print("⚠️ AvgToes: Not enough common timestamps.")
        return ("AvgToes", None, None)

    left_dict = dict(left)
    right_dict = dict(right)

    avg_distances = [(t, (left_dict[t] + right_dict[t]) / 2) for t in common_times]
    times = np.array([t for t, _ in avg_distances])
    dists = np.array([d for _, d in avg_distances])

    A = np.vstack([times, np.ones_like(times)]).T
    m, b = np.linalg.lstsq(A, dists, rcond=None)[0]
    y_pred = m * times + b
    ss_res = np.sum((dists - y_pred) ** 2)
    ss_tot = np.sum((dists - np.mean(dists)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    if verbose:
        print(f"✅ AvgToes: velocity = {m:.4f} m/s | R² = {r2:.4f}")

    return ("AvgToes", m, r2)
