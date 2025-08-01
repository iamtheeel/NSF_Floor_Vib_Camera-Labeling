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
dataFreqRange_hz = [0,0] # will want this later

#dir = 'StudentData/25_06_03/Subject_1'
#dataFile = "data/Yoko_s3_3.hdf5"
#dataFile = "Kera_2.hdf5"
dir = r"E:\STARS"
dataFile = r"Yoko_s3_Run1.hdf5"

dirFile = f"{dir}/{dataFile}"

# What data are we interested in
chToPlot = [1, 2, 3, 4, 5, 6, 7, 10]

# Libraries needed
import csv
from datetime import datetime
from datetime import timedelta
import h5py                             # For loading the data : pip install h5py
import matplotlib.pyplot as plt         # For plotting the data: pip install matplotlib
import numpy as np                      # cool datatype, fun matix stuff and lots of math (we use the fft)    : pip install numpy==1.26.4
                                        # The footstep cwt requires an older verstion of numpy

### 
# Functions
###

## Data Loaders
def print_attrs(name, obj): #From Chatbot
        print(f"\n📂 Path: {name}")
        for key, val in obj.attrs.items():
            print(f"  🔧 Attribute - {key}: {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"  📊 Dataset - Shape: {obj.shape}, Dtype: {obj.dtype}")

def get_peram(perams, peramName:str, asStr=False):
    if asStr:
        peram_value= perams[perams['parameter'] == peramName.encode()]['value'][0].decode('utf8')
    else:
        peram_value= perams[perams['parameter'] == peramName.encode()]['value'][0] 
    units_value = perams[perams['parameter'] == peramName.encode()]['units'][0].decode('utf-8')
    print(f"{peramName}: {peram_value} {units_value}")

    return peram_value, units_value

def get_perams(perams, peramName:str, asType='dateTime'):
    values = [
        #row['value'].decode()
        datetime.fromtimestamp(float(row['value'].decode()))
        for row in perams
            if row['parameter'] == peramName.encode()
    ]

    return values 
def loadPeramiters(dataFile):
    with h5py.File(dataFile, 'r') as h5file:
        filePerams = h5file['experiment/general_parameters'][:]

    #Extract the data capture info from the file
    dataCapRate_hz, dataCapUnits = get_peram(filePerams, 'fs')
    recordLen_s, _ = get_peram(filePerams, 'record_length')
    preTrigger_s, _ = get_peram(filePerams, 'pre_trigger')

    print(filePerams.dtype.names)   # Show the peramiter field names
    print(f"experiment/general_parameters: {filePerams}")          #Show the peramiters

    # Now that we know which is the timepoints
    print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {recordLen_s} seconds long")

    if dataFreqRange_hz[1] == 0: dataFreqRange_hz[1] = dataCapRate_hz/2
    if dataTimeRange_s[1] == 0: dataTimeRange_s[1] = int(recordLen_s)


    return dataCapRate_hz, recordLen_s, preTrigger_s 

def loadData(dataFile, trial=-1):
    """
    Loads the data form an hdf version 5 file

    Args:
        dataFile: String of the data file name and location

    Returns:
        numpy: data 
        int: Data Capture Rate

    """
    print(f"Loading file: {dataFile}")

    with h5py.File(dataFile, 'r') as h5file:
        #if trial == 0: h5file.visititems(print_attrs)

        #filePerams = h5file['experiment/general_parameters'][:]
        if trial >= 0:
            dataFromFile = h5file['experiment/data'][trial,:,:] #Load trial in question
            runPerams = h5file['experiment/specific_parameters']#Load all the rows of data to the block, will not work without the [:]
            triggerTimes, _ = get_peram(runPerams, 'triggerTime', asStr=False)
            triggerTimes = next(
                                row['value'] for row in runPerams
                                if row['parameter'] == b'triggerTime' and row['id'] == trial
                                ).decode() #Get from string
            triggerTimes = datetime.fromtimestamp(float(triggerTimes))
            print(f"Loaded trial: {trial}")
        elif trial == -1: # Load the whole thing
            dataFromFile = h5file['experiment/data'][:] #Load all the rows of data to the block, will not work without the [:]
            runPerams = h5file['experiment/specific_parameters']#Load all the rows of data to the block, will not work without the [:]
            triggerTimes = get_perams(runPerams, 'triggerTime', asType='dateTime')
        # Otherwize, we are just after the peramiters
    # Done getting the file link

    if trial <=0:
        #print(filePerams.dtype.names)   # Show the peramiter field names
        #print(f"experiment/general_parameters: {filePerams}")          #Show the peramiters
        # Look at the shape of the data
        print(f"Data type: {type(dataFromFile)}, shape: {dataFromFile.shape}")

    # We happen to know that:
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

    #if dataTimeRange_s[1] == 0: #If 0, set to length of data
    #    timeLen_s   = (numTimePts)/dataCapRate_hz # How far apart is each time point
    #    dataTimeRange_s[1] = timeLen_s
    #    print(f"dataTimeRange: {dataTimeRange_s}")

    return dataFromFile, triggerTimes

def sec_to_hms(sec):
    return str(timedelta(seconds=int(sec)))

## Data slicers
def sliceTheData(dataBlock:np, chList, timeRange_sec, trial=-1):
    """
    Cuts the data by:
        ch
    
    Args:
        dataBlock: the raw data [Trial, ch, timePoints]
        trial: if -1, then the data is already pre-cut for trial

    Returns:
        numpy: the cut data
    """

    # The ch list
    chList_zeroIndexed = [ch - 1 for ch in chList]  # Convert to 0-based indexing
    print(f"ChList index: {chList_zeroIndexed}")

    # The time range
    dataPoint_from = int(timeRange_sec[0]*dataCapRate_hz)
    dataPoint_to = int(timeRange_sec[1]*dataCapRate_hz)
    print(f"Data Point Range: {dataPoint_from}:{dataPoint_to} at {dataCapRate_hz} hz")

    # Ruturn the cut up data
    if trial > 0:
        return dataBlock[trial, chList_zeroIndexed, dataPoint_from:dataPoint_to]
    else:
        return dataBlock[chList_zeroIndexed, dataPoint_from:dataPoint_to]


## Data Plottters
def dataPlot_2Axis(dataBlockToPlot:np, plotChList, trial:int, xAxisRange, yAxisRange, dataRate:int=0, 
                   domainToPlot:str="time", logX=False, logY=False, title="", save=""):
    """
    Plots the data in 2 axis (time or frequency domain)

    Args:
        dataBlockToPlot (Numpy): The data to be plotted [ch, timepoints]

    Returns:
        Null
    """
    numTimePts = dataBlockToPlot.shape[1]
    if domainToPlot == "time":
        start_time_sec = (triggerTimes - triggerTimes.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        xAxis_data = start_time_sec + np.arange(numTimePts) / dataCapRate_hz
        xAxis_str = f"Clock Time"
        xAxisUnits_str = "(s since midnight)"

    if domainToPlot == "freq":
        xAxis_data = np.fft.rfftfreq(numTimePts, d=1.0/dataRate)
        xAxis_str = f"Frequency"
        xAxisUnits_str = "(Hz)"
    title_str = f"{xAxis_str} Domain plot of trial: {trial} ch: {plotChList}{title}, Acceleration (g)"

    fig, axs = plt.subplots(len(plotChList)) #Make the subplots for how many ch you want
    fig.suptitle(title_str)

    # Make room for the title, axis lables, and squish the plots up against eachother
    fig.subplots_adjust(top = 0.95, bottom = 0.1, hspace=0, left = 0.1, right=0.99) # Mess with the padding (in percent)

    for i, thisCh in enumerate(plotChList):  # Enumerate will turbo charge the forloop, give the value and the idex
        # Plot the ch data
        timeD_data = dataBlockToPlot[i,:]  #Note: Numpy will alow negitive indexing (-1 = the last row)
        if domainToPlot == "time":
            yAxis_data = timeD_data
        if domainToPlot == "freq":
            # Calculate the fft
            # Apply a hanning window to minimize spectral leakage
            window = np.hanning(len(timeD_data))
            timeD_data = timeD_data - np.mean(timeD_data)  # Center the signal before FFT
            timeD_data_windowed = window*timeD_data
            timeD_data_windowed /= np.sum(window) / len(window)  # Normalize
            freqD_data = np.fft.rfft(timeD_data_windowed) # Real value fft returns only below the nyquist
                                                          # The data is returned as a complex value
            freqD_mag = np.abs(freqD_data)                  # Will only plot the magnitude
            yAxis_data = freqD_mag

        print(f"Ch {thisCh} Min: {np.min(yAxis_data)}, Max: {np.max(yAxis_data)}, Mean: {np.mean(yAxis_data)}")
        axs[i].plot(xAxis_data, yAxis_data)
    
        # Set the Axis limits and scale
        #axs[i].set_xlim(xAxisRange) 
        axs[i].set_xlim([xAxis_data[0], xAxis_data[-1]])
        axs[i].set_ylim(yAxisRange)
        if logX: axs[i].set_xscale('log')  # Set log scale
        if logY: axs[i].set_yscale('log')  # Set log scale

        # Label the axis
        axs[i].set_ylabel(f'Ch {plotChList[i]}', fontsize=8)
        if i < len(plotChList) - 1:
            axs[i].set_xticklabels([]) # Hide the xTicks from all but the last

    #Only show the x-axis on the last plot
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].set_xlabel("Time (s)")
    axs[-1].set_xticks(np.linspace(xAxis_data[0], xAxis_data[-1], 10))
    axs[-1].set_xticklabels([sec_to_hms(s) for s in np.linspace(xAxis_data[0], xAxis_data[-1], 10)], rotation=45)

    #plt.savefig(f"images/{save}_{domainToPlot}_trial-{trial}.jpg")
    #plt.close()


    csv_path = r"E:\STARS\Trail_Data"
    csv_filename = f"trial_{trial}_graph_shoe_output.csv"
    csv_file = f"{csv_path}/{csv_filename}"
    with open(csv_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header: Time, Ch1, Ch2, ...
        header = ["Time"] + [f"Ch{ch}" for ch in plotChList]
        writer.writerow(header)
        # Write data: each row is [time, ch1_val, ch2_val, ...]
        for t_idx in range(len(xAxis_data)):
            row = [xAxis_data[t_idx]]
            for ch_idx in range(len(plotChList)):
                row.append(dataBlockToPlot[ch_idx, t_idx])
            writer.writerow(row)
    return xAxis_data # Save for later use

#### Do the stuff
# Load the data 
# Get the peramiters once
dataCapRate_hz, recordLen_s, preTrigger_s = loadPeramiters(dataFile=dirFile) 
#print(triggerTime[0].strftime("%Y-%m-%d %H:%M:%S.%f"))
#exit()

#trialList = [0, 1, 2, 7]
trialList = [14-1]
#trialList = [7, 9, 10, 11]
#for trial in range(20): # Cycle through the trials
for i, trial in enumerate(trialList): # Cycle through the trials

    print(f"Running Trial: {trial}")
    dataBlock_numpy, triggerTimes = loadData(dataFile=dirFile, trial=trial)
    #dataBlock_numpy, dataCapRate_hz, recordLen_s, preTrigger_s, triggerTimes = loadData(dataFile=dirFile, trial=trial)
    print(f"Trigger Time: {triggerTimes.strftime("%Y-%m-%d %H:%M:%S.%f")}")
    print(f"max: {np.max(dataBlock_numpy[3,5])}, mean: {np.mean(dataBlock_numpy)}")
    # Get the parts of the data we are interested in:
    print(f"Data len pre-cut: {dataBlock_numpy.shape}")
    dataBlock_sliced = sliceTheData(dataBlock=dataBlock_numpy, trial=-1, chList=chToPlot, timeRange_sec=dataTimeRange_s) # -1 if the data is already with the trial
    #dataBlock_sliced = sliceTheData(dataBlock=dataBlock_numpy, trial=trial, chList=chToPlot, timeRange_sec=dataTimeRange_s)
    print(f"Data len: {dataBlock_sliced.shape}")

    # Plot the data in the time domain
    timeYRange = 0.01
    #timeYRange = np.max(np.abs(dataBlock_sliced))
    timeSpan = dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=trial, 
                              xAxisRange=dataTimeRange_s, yAxisRange=[-1*timeYRange, timeYRange], domainToPlot="time", save="original")
    plt.show() # Open the plot(s)