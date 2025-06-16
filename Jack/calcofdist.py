import numpy as np
from scipy.optimize import curve_fit

def fit_inverse_model(y_pixels, z_distances, initial_guess=None):
    # Define the model: Z = a / (y + b) + c
    def inverse_model(y, a, b, c):
        return a / (y + b) + c

    if initial_guess is None:
        initial_guess = [-20, 2300, 0]

    # Fit the model to the data using curve_fit
    params, _ = curve_fit(inverse_model, y_pixels, z_distances, p0=initial_guess, maxfev=5000)
    a, b, c = params
    print(f"Fitted equation: Z(y) = {a:.4f} / (y + {b:.4f}) + {c:.4f}")
    return a, b, c

# Input dataset: pixel Y-values (shifted) and real distances
y_pixels = np.array([290, 310, 330, 353, 379, 407, 441, 474, 523, 574, 633, 704, 791, 901, 1043, 1227, 1481])
z_distances = np.array([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4])

# Example usage:
fit_inverse_model(y_pixels, z_distances)
