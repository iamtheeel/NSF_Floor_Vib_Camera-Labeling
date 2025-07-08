####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Playing with pytorch
# make a dataloader and start with tensors
####

import os
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import pandas as pd # For loading the csv
import matplotlib.pyplot as plt # Ploting


class SlidingWindowHeelDataset(Dataset):
    def __init__(self, folder_path, window_size=64, stride=32):
        self.time = []
        self.samples = []
        self.labels = []

        for fileName in os.listdir(folder_path):
            if not fileName.endswith('.csv'): continue # Skip over non csv files

            # Load the csv file
            print(f"Loading: {fileName}")
            path = os.path.join(folder_path, fileName)
            fileData = pd.read_csv(path)

            # Get the left and right foot data
            left = fileData['LeftHeel (m)'].values
            right = fileData['RightHeel (m)'].values
            time = fileData['Time (sec)'].values
            # Convert to tensors
            left_tensor = torch.tensor(left, dtype=torch.float32)
            right_tensor = torch.tensor(right, dtype=torch.float32)
            time_tensor = torch.tensor(time, dtype=torch.float32)

            # Sliding window from left foot
            for i in range(0, len(left_tensor) - window_size + 1, stride):
                window = left_tensor[i:i + window_size]
                time_window = time_tensor[i:i + window_size]
                self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
                self.labels.append(0)  # 0 = Left
                self.time.append(time_window)

            # Sliding window from right foot
            for i in range(0, len(right_tensor) - window_size + 1, stride):
                window = right_tensor[i:i + window_size]
                time_window = time_tensor[i:i + window_size]
                self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
                self.labels.append(1)  # 1 = Right
                self.time.append(time_window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]  # shape: (window_size, 2)

# Make a Dataset
dataset = SlidingWindowHeelDataset("../StudentData/25_06_18/", window_size=60, stride=5)
# Change the window/stride to seconds
print(f"Total samples in dataset: {len(dataset)}")

label_names = {0: "Left Foot", 1: "Right Foot"}
storedata = 0
rightfoot = True

'''
for i, (window, label) in enumerate(dataset):
    if rightfoot == True:
        if label != 1: #skip to right foot
            continue 
    sampling_rate = 30  # Set this to your actual sampling rate (Hz)
    x_sec = (np.arange(len(window)) / sampling_rate)  # seconds
    plt.plot(x_sec, window)
    if label == 0:
        plt.title(f"{label_names[label]} sample starting at {i*1/30:.2f} secs")
    if label == 1:
        if storedata == 0:
            storedata = i
        plt.title(f"{label_names[label]} sample starting at {(i*1/30-storedata*1/30):.2f} secs")
    plt.xlabel("time (sec)")
    plt.ylabel("distance from cam (m)")
    plt.show()
    # Plot the data
    # Change the x-axis to miliseconds
'''


train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, (batch_window, batch_label) in enumerate(train_loader):
    sampling_rate = 30  # Hz
    for j in range(batch_window.shape[0]):  # Loop over batch
        window = batch_window[j].squeeze()  # shape: (window_size,)
        label = batch_label[j].item()
        x_sec = np.arange(len(window)) / sampling_rate
        #print(self)
        plt.plot(x_sec, window)
        plt.title(f"{label_names[label]} sample in batch {i}, index {j}")
        plt.xlabel("time (sec)")
        plt.ylabel("distance from cam (m)")
        plt.show()

    # Plot the windows in the batch
    

# Make a test/train dataset
# Hint use torch.utils.data.random_split and make two datasets from one

