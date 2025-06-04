###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Minimum Case DataLoad, time domain
###

### Settings
dataTimeRange_s = [0, 0] # [0 0] for full dataset

#dataFile = "data/Yoko_s3_3.hdf5"
dataFile = "data/Yoko_s3_1.hdf5"

# What data are we interested in
chToPlot = [1, 2, 3, 4 ,5]

# Librarys needed
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
        if trial == 0:
            h5file.visititems(print_attrs)

        filePerams = h5file['experiment/general_parameters'][:]
        if trial >= 0:
            #print(f"Loading trial: {trial}")
            dataFromFile = h5file['experiment/data'][trial,:,:] #Load trial in question
        elif trial == -1: # Load the whole thing
            print(f"Loading the full dataset")
            dataFromFile = h5file['experiment/data'][:] #Load all the rows of data to the block, will not work without the [:]
        # Otherwize, we are just after the peramiters

    #Extract the data capture rate from the file
    # Data cap rate is the first entery (number 0)
    dataCapRate_hz =filePerams[0]['value']  # Some files needs decode, others can't have it
    #dataCapRate_hz =int(filePerams[0]['value'].decode('utf-8'))  # Some files needs decode, others can't have it

    dataCapUnits = filePerams[0]['units'].decode('utf-8')
    if trial <=0:
        print(f"experiment/general_parameters: {filePerams}")          #Show the peramiters
        print(filePerams.dtype.names)   # Show the peramiter field names
        print(f"Data Cap Rate ({filePerams[0]['parameter'].decode('utf-8')}): {dataCapRate_hz} {dataCapUnits}")
    
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

    if trial <=0:
        # Now that we know which is the timepoints
        timeLen_s   = (numTimePts-1)/dataCapRate_hz # How far apart is each time point
        if dataTimeRange_s[1] == 0: dataTimeRange_s[1] = int(timeLen_s)
        #if dataFreqRange_hz[1] == 0: dataFreqRange_hz[1] = dataCapRate_hz/2
        print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {timeLen_s} seconds long")

    return dataFromFile, dataCapRate_hz

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
        xAxis_data = np.linspace(xAxisRange[0], xAxisRange[1], numTimePts) #start, stop, number of points
        xAxis_str = f"Time"
        xAxisUnits_str = "(s)"

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
        axs[i].set_xlim(xAxisRange) 
        axs[i].set_ylim(yAxisRange)
        if logX: axs[i].set_xscale('log')  # Set log scale
        if logY: axs[i].set_yscale('log')  # Set log scale

        # Label the axis
        axs[i].set_ylabel(f'Ch {plotChList[i]}', fontsize=8)
        if i < len(plotChList) - 1:
            axs[i].set_xticklabels([]) # Hide the xTicks from all but the last

    #Only show the x-axis on the last plot
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].set_xlabel(f"{xAxis_str} {xAxisUnits_str}")

    #plt.savefig(f"images/{save}_{domainToPlot}_trial-{trial}.jpg")
    #plt.close()
    return xAxis_data # Save for later use

#### Do the stuff
# Load the data 
dummyData, dataCapRate_hz = loadData(dataFile=dataFile, trial=0 ) # Just get the peramiters

# 2-22.21-APDM-data.xlsx has 27 enterys, so this is probably the data
# Is this even in the right order???
trialList = [0]

#for trial in range(20): # Cycle through the trials
#for trial in range(dataBlock_numpy.shape[0]): # Cycle through the trials
for i, trial in enumerate(trialList): # Cycle through the trials
    print(f"Running Trial: {trial}")
    dataBlock_numpy, dataCapRate_hz = loadData(dataFile=dataFile, trial=trial)
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