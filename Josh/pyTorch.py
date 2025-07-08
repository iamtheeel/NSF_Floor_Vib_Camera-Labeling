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
import matplotlib.pyplot as plt # Ploting


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

# Make a Dataset
dataset = SlidingWindowHeelDataset("StudentData/25_06_18/", window_size=60, stride=5)
# Change the window/stride to seconds
print(f"Total samples in dataset: {len(dataset)}")

for i, (window, label) in enumerate(dataset):
    print(f"Window {i}: {window.shape}, label: {label}") 
    plt.plot(window)
    plt.show()
    # Plot the data
    # Change the x-axis to miliseconds

exit()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, (batch_window, batch_label) in enumerate(train_loader):
    print(f"Type: {type(batch_window)}, {type(batch_label)}")  
    print(f"Shape: {batch_window.shape}, {batch_label.shape}")
    print(batch_label)

    # Plot the windows in the batch
    

# Make a test/train dataset
# Hint use torch.utils.data.random_split and make two datasets from one

