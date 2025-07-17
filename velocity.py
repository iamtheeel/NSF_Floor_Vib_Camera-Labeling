from collections import defaultdict
import numpy as np

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
