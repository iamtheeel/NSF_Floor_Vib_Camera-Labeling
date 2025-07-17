###
# Jack Capito
# MIC Lab
# Spring, 2025
###
# Scrolling vibration data
###

### Imports ###
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time as pytime
from collections import deque
from scipy.signal import decimate

### Global Settings ###
dataTimeRange_s = [0, 0]  # [0 0] for full dataset
dataFreqRange_hz = [0, 0]
oldData = False
dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_07-10'
dataFile = "Jack_clockTest_interuptVPoll.hdf5"
dirFile = f"{dir}/{dataFile}"
chToPlot = [1]  # Example: plot channels 1, 2, and 5 (1-based index)
target_fps = 1  # Target frames per second for scrolling plot

### Data Loader Functions ###
def print_attrs(name, obj):
    print(f"\n📂 Path: {name}")
    for key, val in obj.attrs.items():
        print(f"  🔧 Attribute - {key}: {val}")
    if isinstance(obj, h5py.Dataset):
        print(f"  📊 Dataset - Shape: {obj.shape}, Dtype: {obj.dtype}")

def get_peram(perams, peramName: str, asStr=False):
    if asStr:
        peram_value = float(perams[perams['parameter'] == peramName.encode()]['value'][0].decode('utf8'))
    else:
        peram_value = perams[perams['parameter'] == peramName.encode()]['value'][0]
    units_value = perams[perams['parameter'] == peramName.encode()]['units'][0].decode('utf-8')
    return peram_value, units_value

def get_perams(perams, peramName: str, asType='dateTime'):
    values = [
        datetime.fromtimestamp(float(row['value'].decode()))
        for row in perams
        if row['parameter'] == peramName.encode()
    ]
    return values

def loadPeramiters(dataFile):
    with h5py.File(dataFile, 'r') as h5file:
        nTrials = h5file['experiment/data'][:].shape[0]
        filePerams = h5file['experiment/general_parameters'][:]
    fs_hz, dataCapUnits = get_peram(filePerams, 'fs', asStr=oldData)
    recordLen_s, _ = get_peram(filePerams, 'record_length', asStr=oldData)
    preTrigger_s, _ = get_peram(filePerams, 'pre_trigger')
    if dataFreqRange_hz[1] == 0: dataFreqRange_hz[1] = fs_hz / 2
    if dataTimeRange_s[1] == 0: dataTimeRange_s[1] = int(recordLen_s)
    return fs_hz, recordLen_s, preTrigger_s, nTrials

def loadData(dataFile, trial=-1):
    with h5py.File(dataFile, 'r') as h5file:
        if trial >= 0:
            dataFromFile = h5file['experiment/data'][trial, :, :]
            runPerams = h5file['experiment/specific_parameters']
            if not oldData:
                triggerTimes = next(
                    row['value'] for row in runPerams
                    if row['parameter'] == b'triggerTime' and row['id'] == trial
                ).decode()
                triggerTimes = datetime.fromtimestamp(float(triggerTimes))
        elif trial == -1:
            dataFromFile = h5file['experiment/data'][:]
            runPerams = h5file['experiment/specific_parameters']
            if not oldData:
                triggerTimes = get_perams(runPerams, 'triggerTime', asType='dateTime')
    return dataFromFile, triggerTimes

def sliceTheData(dataBlock: np.ndarray, chList, timeRange_sec, dataCapRate, trial=-1):
    chList_zeroIndexed = [ch - 1 for ch in chList]
    dataPoint_from = int(timeRange_sec[0] * dataCapRate)
    dataPoint_to = int(timeRange_sec[1] * dataCapRate)
    if trial > 0:
        return dataBlock[trial, chList_zeroIndexed, dataPoint_from:dataPoint_to]
    else:
        return dataBlock[chList_zeroIndexed, dataPoint_from:dataPoint_to]

def downSampleData(data, dataCapRate, downSample):
    nCh, timePoints = data.shape
    example_out = decimate(data[0], downSample, ftype='iir', zero_phase=True)
    downSampled_data = np.empty((nCh, example_out.shape[0]))
    for ch in range(nCh):
        downSampled_data[ch] = decimate(data[ch], downSample, ftype='iir', zero_phase=True)
    return downSampled_data, dataCapRate / downSample


def intervalOutputScroll(dataBlock_sliced, dataCapRate_hz, target_fps=15, channels=None):
    """
    Real-time scrolling plot for multiple channels.
    dataBlock_sliced: shape [1+num_channels, timepoints], first row is time.
    channels: list of channel numbers (1-based) for labeling.
    """
    interval_sec = 5
    n_points = int(interval_sec * dataCapRate_hz)
    total_points = dataBlock_sliced.shape[1]
    time_array = dataBlock_sliced[0, :] - dataBlock_sliced[0, 0]
    channel_data = dataBlock_sliced[1:, :]

    from collections import deque
    import matplotlib.pyplot as plt

    time_window = deque(maxlen=n_points)
    data_windows = [deque(maxlen=n_points) for _ in range(channel_data.shape[0])]

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    for idx, ch in enumerate(channels):
        line, = ax.plot([], [], label=f'Ch {ch}')
        lines.append(line)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vibration")
    ax.set_title("Scrolling Vibration Data (5s window)")
    ax.legend()
    plt.tight_layout()

    for i in range(n_points):
        time_window.append(time_array[i])
        for ch_idx in range(channel_data.shape[0]):
            data_windows[ch_idx].append(channel_data[ch_idx, i])
    for ch_idx, line in enumerate(lines):
        line.set_data(list(time_window), list(data_windows[ch_idx]))
    ax.set_xlim(time_window[0], time_window[-1])
    fig.canvas.draw()
    fig.canvas.flush_events()

    step = max(1, int(dataCapRate_hz // target_fps))
    start_wall_time = pytime.time()

    for i in range(n_points, total_points, step):
        frame_start = pytime.time()
        for j in range(step):
            idx = i + j
            if idx >= total_points:
                break
            time_window.append(time_array[idx])
            for ch_idx in range(channel_data.shape[0]):
                data_windows[ch_idx].append(channel_data[ch_idx, idx])

        if not plt.fignum_exists(fig.number):
            break

        for ch_idx, line in enumerate(lines):
            line.set_data(list(time_window), list(data_windows[ch_idx]))
        ax.set_xlim(time_window[0], time_window[-1])
        fig.canvas.draw()
        fig.canvas.flush_events()

        elapsed_wall = pytime.time() - start_wall_time
        elapsed_data = time_array[i]
        lagTime = (elapsed_data - elapsed_wall)
        frame_time = pytime.time() - frame_start

        # Debug print for each frame
        print(f"Frame {i}: frame_time={frame_time:.4f}s, elapsed_data={elapsed_data:.4f}s, elapsed_wall={elapsed_wall:.4f}s, lagTime={lagTime:.4f}s")

        min_wait = max(0, (1.0 / target_fps) - frame_time)
        if lagTime > min_wait:
            pytime.sleep(lagTime)
        elif min_wait > 0:
            pytime.sleep(min_wait)

    plt.ioff()
    plt.show()

### Main Script ###

fileDataCapRate_hz, recordLen_s, preTrigger_s, nTrials = loadPeramiters(dataFile=dirFile)
dataCapRate_hz = fileDataCapRate_hz
print(f"Data cap rate: {fileDataCapRate_hz} Hz, Record Length: {recordLen_s} sec, pretrigger len: {preTrigger_s}sec, Trials: {nTrials}")

trialList = [1]  # Or use: trialList = list(range(nTrials)) for all trials

for i, trial in enumerate(trialList):
    dataBlock, triggerTime = loadData(dataFile=dirFile, trial=trial)
    num_timepoints = dataBlock.shape[1]
    time_array = np.arange(num_timepoints) / dataCapRate_hz  # start at 0.0

    # Convert to 0-based indices for numpy
    channels_idx = [ch - 1 for ch in chToPlot]
    selected_channels = dataBlock[channels_idx, :]  # shape: [num_channels, timepoints]

    dataBlock_sliced = np.vstack([time_array, selected_channels])
    intervalOutputScroll(dataBlock_sliced, dataCapRate_hz, target_fps=target_fps, channels=chToPlot)
