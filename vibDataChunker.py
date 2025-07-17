###
# Jack Capito
# MIC Lab
# Spring, 2025
###
# Scrolling vibration data class call
###

### Imports ###
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time as pytime
from collections import deque
from scipy.signal import decimate


class vibDataWindow:
    def __init__(self, dir_path, data_file, trial, old_data=False, window=5):
        self.dir = dir_path
        self.dataFile = data_file
        self.dirFile = f"{self.dir}/{self.dataFile}"
        self.trial = trial
        self.chToPlot = [1]
        self.oldData = old_data
        self.window = window
        self.dataTimeRange_s = [0, 0]

    def print_attrs(self, name, obj):
        print(f"\n📂 Path: {name}")
        for key, val in obj.attrs.items():
            print(f"  🔧 Attribute - {key}: {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"  📊 Dataset - Shape: {obj.shape}, Dtype: {obj.dtype}")

    def get_peram(self, perams, peramName: str, asStr=False):
        if asStr:
            peram_value = float(perams[perams['parameter'] == peramName.encode()]['value'][0].decode('utf8'))
        else:
            peram_value = perams[perams['parameter'] == peramName.encode()]['value'][0]
        units_value = perams[perams['parameter'] == peramName.encode()]['units'][0].decode('utf-8')
        return peram_value, units_value

    def get_perams(self, perams, peramName: str):
        return [
            datetime.fromtimestamp(float(row['value'].decode()))
            for row in perams
            if row['parameter'] == peramName.encode()
        ]

    def load_parameters(self):
        with h5py.File(self.dirFile, 'r') as h5file:
            nTrials = h5file['experiment/data'][:].shape[0]
            filePerams = h5file['experiment/general_parameters'][:]
        fs_hz, _ = self.get_peram(filePerams, 'fs', asStr=self.oldData)
        recordLen_s, _ = self.get_peram(filePerams, 'record_length', asStr=self.oldData)
        preTrigger_s, _ = self.get_peram(filePerams, 'pre_trigger')
        if self.dataTimeRange_s[1] == 0:
            self.dataTimeRange_s[1] = int(recordLen_s)
        return fs_hz, recordLen_s, preTrigger_s, nTrials

    def load_data(self, trial=-1):
        with h5py.File(self.dirFile, 'r') as h5file:
            if trial >= 0:
                dataFromFile = h5file['experiment/data'][trial, :, :]
                runPerams = h5file['experiment/specific_parameters']
                triggerTime = None
                if not self.oldData:
                    triggerRaw = next(
                        row['value'] for row in runPerams
                        if row['parameter'] == b'triggerTime' and row['id'] == trial
                    ).decode()
                    triggerTime = datetime.fromtimestamp(float(triggerRaw))
            else:
                dataFromFile = h5file['experiment/data'][:]
                runPerams = h5file['experiment/specific_parameters']
                triggerTime = self.get_perams(runPerams, 'triggerTime')
        return dataFromFile, triggerTime

    def slice_data(self, dataBlock, chList, timeRange_sec, dataCapRate, trial=-1):
        chList_zeroIndexed = [ch - 1 for ch in chList]
        from_idx = int(timeRange_sec[0] * dataCapRate)
        to_idx = int(timeRange_sec[1] * dataCapRate)
        if trial >= 0:
            return dataBlock[trial, chList_zeroIndexed, from_idx:to_idx]
        else:
            return dataBlock[chList_zeroIndexed, from_idx:to_idx]

    def time_to_seconds(self, t):
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

    def show_data_at_time(self, target_time_str, dataBlock, trigger_time, trial_num, dataCapRate, preTrigger=0):
        if isinstance(target_time_str, str):
            target_time_obj = datetime.strptime(target_time_str, "%H:%M:%S.%f").time()
        else:
            target_time_obj = target_time_str

        trigger_sec = self.time_to_seconds(trigger_time.time()) - preTrigger
        target_sec = self.time_to_seconds(target_time_obj)
        time_offset_sec = target_sec - trigger_sec - self.window

        if time_offset_sec < 0:
            if time_offset_sec > -self.window:
                self.window += time_offset_sec
                time_offset_sec = 0
            else:
                print(f"Requested time {target_time_str} is out of bounds.")
                return

        print(f"Jumping to {time_offset_sec:.3f}s after trigger for trial {trial_num}")
        sliced_data = self.slice_data(
            dataBlock, self.chToPlot, [time_offset_sec, time_offset_sec + self.window], dataCapRate, trial=-1
        )

        time_axis = np.linspace(time_offset_sec, time_offset_sec + self.window, sliced_data.shape[1])
        plt.figure()
        for i, ch in enumerate(self.chToPlot):
            plt.plot(time_axis, sliced_data[i], label=f"Ch {ch}")
        plt.title(f"Trial {trial_num} - Time {target_time_str}")
        plt.xlabel("Time (s)")
        plt.ylabel("Vibration (gs)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    #external call
    def vib_get(self, time, trialList, distanceFromCam, hhmmss=False, debug=False):
        fs_hz, recordLen_s, preTrigger_s, nTrials = self.load_parameters()
        if debug:
            print(f"Data cap rate: {fs_hz} Hz, Record Length: {recordLen_s}s, Pre-trigger: {preTrigger_s}s, Trials: {nTrials}")

        for chunk in time:
            if debug:
                print(f"Trial {chunk['trial']} at {chunk['time']}")
            dataBlock_numpy, triggerTime = self.load_data(trial=chunk['trial'])
            self.show_data_at_time(
                target_time_str=chunk['time'],
                dataBlock=dataBlock_numpy,
                trigger_time=triggerTime,
                trial_num=chunk(self.trialToPlot),
                dataCapRate=fs_hz,
                preTrigger=preTrigger_s
            )


            
    """def vibGet(self, timeChunk, dataFile, trialList, hhmmss = False, dataFeedback = False, chunkTest = False):
        '''vibGet arguments:

            **timeChunk**
                input the time in seconds since midnight to access the window
            **dataFile**
                the location of the directory of the trials desired
            **trialList**
                the trial desired to pull from for display
            **hhmmss**
                set to true if in [hh:mm:ss:micross] format
            **dataFeedback**
                Debugging print
            **chunkTest**
                Debugging print
        '''
        fileDataCapRate_hz, recordLen_s, preTrigger_s, nTrials = self.loadPeramiters(dataFile)
        dataCapRate_hz = fileDataCapRate_hz
        if dataFeedback == True:
            print(f"Data cap rate: {fileDataCapRate_hz} Hz, Record Length: {recordLen_s} sec, pretrigger len: {preTrigger_s}sec, Trials: {nTrials}")

        for chunk in timeChunk:
            if chunkTest == True:  
                print(f"Trial {chunk['trial']} at {chunk['time']}")
            dataBlock_numpy, triggerTime = self.loadData(dataFile=dirFile, trial=chunk['trial'])
            showDataAtTime(
                target_time_str=chunk['time'], dataBlock=dataBlock_numpy, trigger_time=triggerTime,
                chList=chToPlot, dataCapRate=dataCapRate_hz, trial_num=chunk['trial'], window_s=window, preTrigger=preTrigger_s)


        return"""