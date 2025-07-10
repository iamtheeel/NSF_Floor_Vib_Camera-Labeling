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

import pandas as pd # For loading the csv
import matplotlib.pyplot as plt # Plotting

class SlidingWindowHeelDataset(Dataset):
    def __init__(self, folder_path, window_size=64, stride=32):
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
            # Convert to tensors
            left_tensor = torch.tensor(left, dtype=torch.float32)
            right_tensor = torch.tensor(right, dtype=torch.float32)

            # Sliding window from left foot
            for i in range(0, len(left_tensor) - window_size + 1, stride):
                window = left_tensor[i:i + window_size]
                self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
                self.labels.append(0)  # 0 = Left

            # Sliding window from right foot
            for i in range(0, len(right_tensor) - window_size + 1, stride):
                window = right_tensor[i:i + window_size]
                self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
                self.labels.append(1)  # 1 = Right


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]  # shape: (window_size, 2)

# Parameters
FPS = 30.0
frame_duration = 1.0 / FPS
window_size_sec = 5

#Dataset
dataset = SlidingWindowHeelDataset("StudentData/25_06_18/", int(FPS*window_size_sec), int(FPS))#1 second
# Change the window/stride to seconds
print(f"Total samples in dataset: {len(dataset)}")

for i, (window, label) in enumerate(dataset):
    print(f"Window {i}: {window.shape}, label: {label}") 

    # Plot the data
    # Change the x-axis to milliseconds

for i in range(5):  # Only plot first 5 pairs (left + right)
    left_window, left_label = dataset[i * 2]     # Left = label 0
    right_window, right_label = dataset[i * 2 + 1]  # Right = label 1

    x_sec = torch.arange(left_window.shape[0]) * frame_duration #time in seconds, number of frames in window, creates 1D tensor

    plt.plot(x_sec, left_window.squeeze().numpy(), label="Left Heel") #removes dimensions of size 1 from tensor; numpy is for plotting
    plt.plot(x_sec, right_window.squeeze().numpy(), label="Right Heel")

    plt.title(f"Sliding Window #{i}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heel Distance (m)") 
    plt.legend()
    plt.grid(True)
    plt.show()
quit()

train_loader = DataLoader(dataset, batch_size=32, shuffle=True) #32 samples per batch

    # Plot the windows in the batch
# === Plotting batch windows (first batch only) ===
for i, (batch_window, batch_label) in enumerate(train_loader): #goes through batch by batch
    print(f"Type: {type(batch_window)}, {type(batch_label)}")  
    print(f"Shape: {batch_window.shape}, {batch_label.shape}")  # shape: (batch_size, window_size, 1)

    window_size = int(5*FPS) #5 seconds window size (x axis)
    stride = int(1*FPS) #1 window, 1 second increment (stride)
    x_sec = torch.arange(batch_window.shape[1]) * frame_duration  # window size = 60
    # Create dataset using 5-second window and 1-second stride
    dataset = SlidingWindowHeelDataset("StudentData/25_06_18/", window_size=window_size, stride=stride)

    for j in range(5):  # Plot first 5 windows in the batch
        plt.plot(x_sec, batch_window[j].squeeze().numpy(),
                 label="Left Heel" if batch_label[j] == 0 else "Right Heel")
        plt.title(f"Batch {i}, Sample {j} (Label: {'Left' if batch_label[j] == 0 else 'Right'})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heel Distance (m)")
        plt.legend()
        plt.grid(True)
        plt.show()

    break  # Only plot the first batch


# Make a test/train dataset
# Hint use torch.utils.data.random_split and make two datasets from one

from torch.utils.data import random_split

# === Create train/test split ===
train_ratio = 0.8  # 80% train, 20% test - validate on 20% of the data (validation is the other)
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# === Create DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
