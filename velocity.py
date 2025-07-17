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
def calculate_avg_heel_velocity(distance_data, lastDataPoint, nPoints, verbose=True): #verbose controls whether the function prints progress or result messages to the console.
                                                              #If true, it will print messages; if false, it will not.
    left_m = distance_data[lastDataPoint].get("LeftHeel_Dist", [])
    right_m = distance_data[lastDataPoint].get("RightHeel_Dist", [])

    # Align timestamps
    times_left = {t for t, _ in left_m}
    times_right = {t for t, _ in right_m}
    common_times = sorted(times_left & times_right) #only compare Left and Right distances from the same video frames.

    if len(common_times) < 2:
        if verbose:
            print("⚠️ AvgHeels: Not enough common timestamps.")
        return ("AvgHeels", None, None)

    left_dict = dict(left_m)
    right_dict = dict(right_m)

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
# for how long of a window
# what is our fps
def calculate_avg_landMark_velocity(landmark_data, left, right, curentFrame, nPoints, verbose=True):
    seconds_np = np.array([entry["seconds_sinceMid"] for entry in landmark_data[curentFrame-nPoints:curentFrame]])
    left_np = np.array([entry[left] for entry in landmark_data[curentFrame-nPoints:curentFrame]])
    right_np = np.array([entry[right] for entry in landmark_data[curentFrame-nPoints:curentFrame]])


    # lstsq needs multi dim for times. We have 2 outputs m, b, so we need 2 inputs: linear algibra
    A = np.vstack([seconds_np, np.ones_like(seconds_np)]).T  # Shape becomes (30, 2) 
    #print(f"shapes seconds: {seconds_np.shape}, left_np: {left_np.shape}, right: {right_np.shape}, A: {A.shape}")
    m_l, b_l = np.linalg.lstsq(A, left_np, rcond=None)[0] #rcond: precision, [0]: just give me the m and b (solution), don't bother with: residuals, rank, singular_values
    m_r, b_r = np.linalg.lstsq(A, right_np, rcond=None)[0] #rcond: precision, [0]: just give me the m and b (solution), don't bother with: residuals, rank, singular_values

    m = (m_l +m_r)/2 # Average speeds of left and right
    #TODO: Do we need the R^2?

    if verbose:
        print(f"✅ velocity left{m_l:.3f}, right: {m_r:.3f}, ave: {m:.3f}")

    return m

    left_m = landmark_data[curentFrame].get(left, [])
    right_m = landmark_data[curentFrame].get(right, [])




    # Align timestamps
    times_left = {t for t, _ in left_m}
    times_right = {t for t, _ in right_m}
    common_times = sorted(times_left & times_right)

    if len(common_times) < 2:
        if verbose:
            print("⚠️ AvgToes: Not enough common timestamps.")
        return ("AvgToes", None, None)

    left_dict = dict(left_m)
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
