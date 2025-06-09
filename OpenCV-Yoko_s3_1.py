import h5py
import matplotlib
matplotlib.use('Agg')  # Use a backend that supports rendering to image buffer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2

# === SETTINGS ===
dataFile = "/Volumes/MY PASSPORT/Stars_day1Data/Yoko_s3_1.hdf5"
trial = 0
chToPlot = [1, 2, 3, 4, 5]
dataTimeRange_s = [0, 30]  # full range

# === LOAD DATA ===
def load_data(dataFile, trial=0):
    print("Loading file:", dataFile)
    with h5py.File(dataFile, 'r') as h5file:
        params = h5file['experiment/general_parameters'][:]
        rate = int(params[0]['value'])  # Hz
        data = h5file['experiment/data'][trial, :, :]  # [channels, timepoints]
    return data, rate

# === SLICE CHANNELS + TIME RANGE ===
def slice_data(dataBlock, rate, chList, timeRange_s):
    ch_idx = [ch - 1 for ch in chList]  # 0-based
    start = int(timeRange_s[0] * rate)
    end = int(timeRange_s[1] * rate)
    return dataBlock[ch_idx, start:end]

# === PLOT AND EXPORT IMAGE TO OPENCV ===
def plot_to_opencv(dataBlock, chList, trial, timeRange_s, yAxisRange):
    fig, axs = plt.subplots(len(chList), figsize=(10, 8), sharex=True)
    FigureCanvas(fig)  # Attach Agg canvas for rendering

    timeAxis = np.linspace(timeRange_s[0], timeRange_s[1], dataBlock.shape[1])

    for i, ch_data in enumerate(dataBlock):
        axs[i].plot(timeAxis, ch_data)
        axs[i].set_ylabel(f"Ch {chList[i]}")
        axs[i].set_ylim(yAxisRange)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Trial {trial} Sensor Data")

    # Save to RGB buffer
    fig.tight_layout()
    fig.canvas.draw()
    img_rgb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img_rgb = img_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img_rgb

# === MAIN PIPELINE ===
dataBlock, rate = load_data(dataFile, trial)
dataSliced = slice_data(dataBlock, rate, chToPlot, dataTimeRange_s)

print(f"Sampling Rate: {rate} Hz")
print(f"Data Shape (after slice): {dataSliced.shape}")

yRange = 0.01
img = plot_to_opencv(dataSliced, chToPlot, trial, dataTimeRange_s, yAxisRange=[-yRange, yRange])

# === SAVE WITH OPENCV ===
outputPath = "/Volumes/MY PASSPORT/Stars_day1Data/opencv_trial_0.png"
cv2.imwrite(outputPath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"✅ Frame saved to: {outputPath}")
