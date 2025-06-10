import numpy as np

# Example data
y_values = np.array([284, 305, 361, 428, 468, 516.5, 569, 630, 702, 891.5, 1016.5, 1175, 1383.5])
distances = np.array([59.25, 56.25, 50.25, 47.25, 44.25, 42.25, 48.25, 35.25, 32.25, 26.25, 23.25, 20.25, 17.25])


def exponential_fit(x_vals, y_vals):
    x = np.array(distances)
    y = np.array(y_values)

    # Transform y to ln(y)
    ln_y = np.log(y)

    # Fit ln_y = b*x + ln(a) using polyfit (degree 1)
    b, ln_a = np.polyfit(x, ln_y, 1)

    a = np.exp(ln_a)

    # Return the exponential equation as a string
    equation = f"y = {a:.4f} * e^({b:.4f}x)"
    return equation

eqn = exponential_fit(distances, y_values)
print(eqn)
