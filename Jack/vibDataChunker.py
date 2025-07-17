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
oldData = False
dir = r'C:\Users\notyo\Documents\STARS\StudentData\25_07-10'
dataFile = "Jack_clockTest_interuptVPoll.hdf5"
dirFile = f"{dir}/{dataFile}"
chToPlot = [1]  # Example: plot channels 1, 2, and 5 (1-based index)
target_fps = 1  # Target frames per second for scrolling plot

# Add manual chunk time here (as list of dicts with trial and time strings)
manualTimechunk = [{"trial": 0, "time": "10:45:53.500000"}]
window = 5

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

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def showDataAtTime(target_time_str, dataBlock, trigger_time, chList, dataCapRate, trial_num, window_s, preTrigger=0):
    if isinstance(target_time_str, str):
        target_time_obj = datetime.strptime(target_time_str, "%H:%M:%S.%f").time()
    else:
        target_time_obj = target_time_str

    trigger_sec = time_to_seconds(trigger_time.time()) - preTrigger
    target_sec = time_to_seconds(target_time_obj)

    time_offset_sec = target_sec - trigger_sec - window_s
    if time_offset_sec < 0:
        if time_offset_sec > -window_s:
            time_offset_redefine = time_offset_sec
            window_s = window_s + time_offset_redefine
            time_offset_sec = 0
        else: 
            print(f"Requested time {target_time_str} is out of bounds.")
            return


    print(f"Jumping to {time_offset_sec:.3f}s after trigger for trial {trial_num}")

    sliced_data = sliceTheData(
        dataBlock, chList, [time_offset_sec, time_offset_sec + window_s], dataCapRate, trial=-1
    )

    time_axis = np.linspace(time_offset_sec, time_offset_sec + window_s, sliced_data.shape[1])

    plt.figure()
    for i, ch in enumerate(chList):
        plt.plot(time_axis, sliced_data[i], label=f"Ch {ch}")
    plt.title(f"Trial {trial_num} - Time {target_time_str}")
    plt.xlabel("Time (s)")
    plt.ylabel("Vibration (gs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


### Main Script ###
fileDataCapRate_hz, recordLen_s, preTrigger_s, nTrials = loadPeramiters(dataFile=dirFile)
dataCapRate_hz = fileDataCapRate_hz
print(f"Data cap rate: {fileDataCapRate_hz} Hz, Record Length: {recordLen_s} sec, pretrigger len: {preTrigger_s}sec, Trials: {nTrials}")

trialList = [1]  # Or use: trialList = list(range(nTrials)) for all trials

# Run manual time-based chunks
for chunk in manualTimechunk:
    print(f"Trial {chunk['trial']} at {chunk['time']}")
    dataBlock_numpy, triggerTime = loadData(dataFile=dirFile, trial=chunk['trial'])
    showDataAtTime(
        target_time_str=chunk['time'], dataBlock=dataBlock_numpy, trigger_time=triggerTime,
        chList=chToPlot, dataCapRate=dataCapRate_hz, trial_num=chunk['trial'], window_s=window, preTrigger=preTrigger_s)
