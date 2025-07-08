####
#   Kara-Leah 
#   STARS Summer 2025
#   Dr J Lab
###
# Playing with pytorch
# make a dataloader and start with tensors
####

import os
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd # For loading the csv
import matplotlib.pyplot as plt # Ploting
import numpy as np # For numerical operations

class SlidingWindowHeelDataset(Dataset):
    def __init__(self, folder_path, window_size=64, stride=32):
        self.samples = []
        self.labels = []
        self.file_names = []

        for fileName in os.listdir(folder_path):
            if not fileName.endswith('.csv'): continue # Skip over non csv files
                     # Load the csv file
            print(f"Loading: {fileName}")
            path = os.path.join(folder_path, fileName)
            fileData = pd.read_csv(path)

            # Get the left and right foot data
            left = fileData['LeftHeel (m)'].values
            right = fileData['RightHeel (m)'].values
            print(f"Left: {len(left)}, Right: {len(right)}")
            # Convert to tensors
            left_tensor = torch.tensor(left, dtype=torch.float32)
            right_tensor = torch.tensor(right, dtype=torch.float32)

            # Sliding window from left foot
            for i in range(0, len(left_tensor) - window_size + 1, stride):
                window = left_tensor[i:i + window_size]
                self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
                self.labels.append(0)  # 0 = Left
                self.file_names.append(fileName)  # Store the file name
            

            # Sliding window from right foot
            for i in range(0, len(right_tensor) - window_size + 1, stride):
                window = right_tensor[i:i + window_size]
                self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
                self.labels.append(1)  # 1 = Right
                self.file_names.append(fileName)  # Store the file name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.file_names[idx]  # shape: (window_size, 2)

# Make a Dataset
fs = 30  # Hz
window_size_s = 5  # 5 seconds
window_size = fs * window_size_s  # Convert to samples
stride_s = 1  # 1 second stride
stride = stride_s * fs  # 30 samples stride

dataset = SlidingWindowHeelDataset("StudentData/25_06_18/", window_size, stride)
# Change the window/stride to seconds
print(f"Total samples in dataset: {len(dataset)}")
print("We're here!")  
for i, (window, label, file) in enumerate(dataset):
    print(f"Window {i}: {window.shape}, label: {label}, file: {file}") 

    # Plot the data
    #window, label = dataset[i]
    window_np = window.squeeze(-1).numpy()  # shape: (window_size,)

    # X-axis in milliseconds
    time_axis = (np.arange(len(window_np)) / fs)  # ms
    # Change the x-axis to miliseconds
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, window_np, label='Heel Position (m)')
    plt.title(f"Sample {i} - Label: {'Left' if label == 0 else 'Right'}")
    plt.xlabel("Time (s)")
    plt.ylabel("Heel Position (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    key = input("Press Enter for next plot, or type q to quit: ")
    if key.lower() == 'q':
        break




train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, (batch_window, batch_label) in enumerate(train_loader):
    print(f"Type: {type(batch_window)}, {type(batch_label)}")  
    print(f"Shape: {batch_window.shape}, {batch_label.shape}")

    # Plot the windows in the batch
