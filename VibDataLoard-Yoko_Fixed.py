###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Minimum Case DataLoad, time domain
###

### Settings
dataTimeRange_s = [0,0] # [0 0] for full dataset

dir = '/Volumes/MY PASSPORT/Stars_day1Data/'
file = 'Yoko_s3_1.hdf5'
filename = f"{dir}{file}"  # Full path to the video file

# What data are we interested in
chToPlot = [11,12,13,14,15,16,17,18,19,20] # 1-20

# Libraries needed
import h5py
import matplotlib.pyplot as plt
import numpy as np

### 
# Functions
###

def print_attrs(name, obj):  # From Chatbot
    print(f"\n📂 Path: {name}")
    for key, val in obj.attrs.items():
        print(f"  🔧 Attribute - {key}: {val}")
    if isinstance(obj, h5py.Dataset):
        print(f"  📊 Dataset - Shape: {obj.shape}, Dtype: {obj.dtype}")


def loadData(dataFile, trial=-1):
    print(f"Loading file: {dataFile}")

    with h5py.File(dataFile, 'r') as h5file:
        if trial == 0:
            h5file.visititems(print_attrs)

        filePerams = h5file['experiment/general_parameters'][:]
        if trial >= 0:
            dataFromFile = h5file['experiment/data'][trial,:,:]
        elif trial == -1:
            print(f"Loading the full dataset")
            dataFromFile = h5file['experiment/data'][:]

    dataCapRate_hz = filePerams[0]['value']
    dataCapUnits = filePerams[0]['units'].decode('utf-8')
    if trial <= 0:
        print(f"experiment/general_parameters: {filePerams}")
        print(filePerams.dtype.names)
        print(f"Data Cap Rate ({filePerams[0]['parameter'].decode('utf-8')}): {dataCapRate_hz} {dataCapUnits}")
        print(f"Data type: {type(dataFromFile)}, shape: {dataFromFile.shape}")

    if trial < 0:
        numTrials = dataFromFile.shape[0]
        numSensors = dataFromFile.shape[1]
        numTimePts = dataFromFile.shape[2]
        print(f"The dataset has: {numTrials} trials, {numSensors} sensors, {numTimePts} timepoints")
    else:
        numSensors = dataFromFile.shape[0]
        numTimePts = dataFromFile.shape[1]
        if trial == 0:
            print(f"The dataset has: {numSensors} sensors, {numTimePts} timepoints")

    if trial <= 0:
        timeLen_s = (numTimePts - 1) / dataCapRate_hz
        if dataTimeRange_s[1] == 0:
            dataTimeRange_s[1] = int(timeLen_s)
        print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {timeLen_s} seconds long")

    return dataFromFile, dataCapRate_hz


def sliceTheData(dataBlock: np, chList, timeRange_sec, dataCapRate_hz, trial=-1):
    chList_zeroIndexed = [ch - 1 for ch in chList]
    print(f"ChList index: {chList_zeroIndexed}")
    dataPoint_from = int(timeRange_sec[0]*dataCapRate_hz)
    dataPoint_to = int(timeRange_sec[1]*dataCapRate_hz)
    if trial > 0:
        return dataBlock[trial, chList_zeroIndexed, dataPoint_from:dataPoint_to]
    else:
        return dataBlock[chList_zeroIndexed, dataPoint_from:dataPoint_to]


def dataPlot_2Axis(dataBlockToPlot: np, plotChList, trial: int, xAxisRange, yAxisRange, dataRate: int = 0, 
                   domainToPlot: str = "time", logX=False, logY=False, title="", save=""):
    numTimePts = dataBlockToPlot.shape[1]
    if domainToPlot == "time":
        xAxis_data = np.linspace(xAxisRange[0], xAxisRange[1], numTimePts)
        xAxis_str = f"Time"
        xAxisUnits_str = "(s)"

    if domainToPlot == "freq":
        xAxis_data = np.fft.rfftfreq(numTimePts, d=1.0/dataRate)
        xAxis_str = f"Frequency"
        xAxisUnits_str = "(Hz)"
    title_str = f"{xAxis_str} Domain plot of trial: {trial} ch: {plotChList}{title}, Acceleration (g)"

    fig, axs = plt.subplots(len(plotChList))
    fig.suptitle(title_str)
    fig.subplots_adjust(top=0.95, bottom=0.1, hspace=0, left=0.1, right=0.99)

    for i, thisCh in enumerate(plotChList):
        timeD_data = dataBlockToPlot[i,:]
        if domainToPlot == "time":
            yAxis_data = timeD_data
        if domainToPlot == "freq":
            window = np.hanning(len(timeD_data))
            timeD_data = timeD_data - np.mean(timeD_data)
            timeD_data_windowed = window*timeD_data
            timeD_data_windowed /= np.sum(window) / len(window)
            freqD_data = np.fft.rfft(timeD_data_windowed)
            freqD_mag = np.abs(freqD_data)
            yAxis_data = freqD_mag

        print(f"Ch {thisCh} Min: {np.min(yAxis_data)}, Max: {np.max(yAxis_data)}, Mean: {np.mean(yAxis_data)}")
        axs[i].plot(xAxis_data, yAxis_data)
        axs[i].set_xlim(xAxisRange) 
        axs[i].set_ylim(yAxisRange)
        if logX: axs[i].set_xscale('log')
        if logY: axs[i].set_yscale('log')
        axs[i].set_ylabel(f'Ch {plotChList[i]}', fontsize=8)
        if i < len(plotChList) - 1:
            axs[i].set_xticklabels([])

    axs[-1].get_xaxis().set_visible(True)
    axs[-1].set_xlabel(f"{xAxis_str} {xAxisUnits_str}")
    return xAxis_data


# === DO THE STUFF ===

dummyData, dataCapRate_hz = loadData(dataFile=filename, trial=0)

# === FIXED: Only loop over existing trials ===
with h5py.File(filename, 'r') as h5file:
    numTrials = h5file['experiment/data'].shape[0]

for trial in range(numTrials):
    print(f"Running Trial: {trial}")
    dataBlock_numpy, dataCapRate_hz = loadData(dataFile=filename, trial=trial)
    print("")

    print(f"Data len pre-cut: {dataBlock_numpy.shape}")
    dataBlock_sliced = sliceTheData(dataBlock=dataBlock_numpy, trial=-1, chList=chToPlot,
                                    timeRange_sec=dataTimeRange_s, dataCapRate_hz=dataCapRate_hz)

    # === Stats ===
    for i in range(dataBlock_numpy.shape[0]):
        y = dataBlock_numpy[i, :]
        y_min = np.min(y)
        y_max = np.max(y)
        y_mean = np.mean(y)
        y_std = np.std(y)
        print(f"Ch{i+1}:  Min:{y_min} Max:{y_max} Mean:{y_mean} Std:{y_std}")

    print(f"Data len: {dataBlock_sliced.shape}")

    timeYRange = 0.01
    timeSpan = dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=trial, 
                              xAxisRange=dataTimeRange_s, yAxisRange=[-1*timeYRange, timeYRange],
                              domainToPlot="time", save="original")
    plt.show()
