####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Playing with pytorch
# make a dataloader and start with tensors
####

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import random

import os
import pandas as pd # For loading the csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # pip install scikit-learn
import matplotlib.pyplot as plt # Ploting
import numpy as np

## Configurations
plotData = False
# Data
dataDir = 'StudentData/25_06_18/expRuns'
sampleFreq_hz =  1/0.033
windowLen_s = 5
strideLen_s = 1
#classes = ["Left", "Right"]
classes = ["Heel", "Toe"]
#classes = ["Left Heel", "Right Heel", "Left Toe", "Right Toe"]

# Training hyperameters
nEpochs = 20
learningRate = 0.001

# rand seed
seed = 1337
random.seed(seed)
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)

### Data Loader ###
class SlidingWindowHeelDataset(Dataset):
    def __init__(self, folder_path, window_size=64, stride=32):
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride

        for fileName in os.listdir(folder_path):
            if not fileName.endswith('.csv'): continue # Skip over non csv files

            # Load the csv file
            print(f"Loading: {fileName}")
            path = os.path.join(folder_path, fileName)
            fileData = pd.read_csv(path)

            # Get the left and right foot data
            h_left = fileData['LeftHeel_Dist'].values
            h_right = fileData['RightHeel_Dist'].values
            t_left = fileData['LeftToe_Dist'].values
            t_right = fileData['RightToe_Dist'].values
            # Convert to tensors
            h_left_tensor = torch.tensor(h_left, dtype=torch.float32)
            h_right_tensor = torch.tensor(h_right, dtype=torch.float32)
            t_left_tensor = torch.tensor(t_left, dtype=torch.float32)
            t_right_tensor = torch.tensor(t_right, dtype=torch.float32)

            # Sliding window from the data
            self.sldWin(h_left_tensor, 0)
            self.sldWin(h_right_tensor, 0)
            self.sldWin(t_left_tensor, 1)
            self.sldWin(t_right_tensor, 1)
        # Done File

    def sldWin(self, data, label):
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]

            window = self.standerdize(window) # Standardize each block to it's self

            self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
            self.labels.append(label) 

    #def normalize(self, dataBlock):

    def standerdize(self, dataBlock):
        mean = torch.mean(dataBlock)
        std = torch.std(dataBlock)
        return (dataBlock - mean)/std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]  # shape: (window_size, 2)
    
### Model ###
class nNet(nn.Module ):
    def __init__(self, input_size, nClasses):
        super(nNet, self).__init__()
        layerSize = 64
        self.fc1 = nn.Linear(input_size, layerSize)
        self.fc2 = nn.Linear(layerSize, layerSize)  
        self.fc3 = nn.Linear(layerSize, layerSize)  
        self.classifyer = nn.Linear(layerSize, nClasses)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (batch, window_len)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.classifyer(x)
        return x

## Plotting ###
def plot_data(i, window, label):
    t = np.linspace(0, windowLen_s, num=len(window), endpoint=False)
    title_str = f"Window {i}: {window.shape}, label: {label}"
    print(title_str) 
    plt.title(title_str)
    plt.plot(t, window)
    plt.xlabel("Time (S)")
    plt.ylabel("Location in Hall (m)")
    plt.show()

## Do the stuff ##
# Make a Dataset
windowLen = int(windowLen_s*sampleFreq_hz)
strideLen = int(strideLen_s*sampleFreq_hz)
dataset = SlidingWindowHeelDataset(dataDir, window_size=windowLen, stride=strideLen)
print(f"Total windows in dataset: {len(dataset)}")

# View  the datasetA
if plotData:
    for i, (window, label) in enumerate(dataset):
        plot_data(i, window, label)
    exit()
## Make dataloaders
train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
train_ds, test_ds = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)
test_loader = DataLoader(test_ds, batch_size=32)

model = nNet(input_size=windowLen, nClasses=len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
loss_fn = nn.CrossEntropyLoss()

## Train 
for epoch in range(nEpochs): 
    model.train() # PUt the model in read write
    total_loss = 0
    correct = 0
    for batch_window, batch_label in train_loader:

        outputs = model(batch_window)        # Run the forward pass
        loss = loss_fn(outputs, batch_label) # Calculate how far off we are

        optimizer.zero_grad()                # zero out the gradiants
        loss.backward()                      # Run the backwards pass
        optimizer.step()                     # Step the optimiser

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_label).sum().item()

    acc = 100. * correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


## ValidatEpoch 500, Loss: 4.8402, Accuracy: 90.92%e
# For Confusion matrix:
all_preds = []
all_labels = []

correct = 0
model.eval() # Put the model in read only
with torch.no_grad():
    for batch_window, batch_label in test_loader:
        outputs = model(batch_window)
        preds = outputs.argmax(dim=1)

        # Look at the results
        correct += (preds == batch_label).sum().item()
        # For confusion matrix
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_label.cpu().numpy())

## Analize
print(f"Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}%")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Plot it
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.grid(False)
plt.show()

# Note, funtionalizeing of dataloader
# Plot loss (how wrong is each guess) and accuracy (how many did we get right)
# Add toes to the classifyer
# Why are the results not the same every time? How to fix?
# Look at the data, what can be done to improve our results?
# Normalize, standardize by window
# Globaly normalize, standardize
# Loss function  -- MSE, RMS..
# Optimizer -- Hill analigy