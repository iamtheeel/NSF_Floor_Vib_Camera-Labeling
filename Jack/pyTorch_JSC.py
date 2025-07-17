####
#   Jack Capito
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

import os
import pandas as pd # For loading the csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # pip install scikit-learn
import matplotlib.pyplot as plt # Ploting
import numpy as np
import time

## Configurations
plotData = False
# Data
dataDir = 'StudentData/25_06_18/expRuns'
sampleFreq_hz =  30#1/0.033
windowLen_s = 5
strideLen_s = 1
#classes = ["None", "Left", "Right"]
classes = ["None", "Heel", "Toe"]
#classes = ["None", "Left Heel", "Right Heel", "Left Toe", "Right Toe"]

# Training hyperparameters
nEpochs = 150
learningRate = 0.001

# Make sure the runs are the same 
seed = 1337
seed = int(time.time()) %1000
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)

### Data Loader ###
class SlidingWindowHeelDataset(Dataset):
    def __init__(self, folder_path, window_size, stride):
        self.samples = []
        self.labels = []
        self.sTime = []
        self.window_size = window_size
        self.stride = stride

        for fileName in os.listdir(folder_path):
            '''
            Loading: sub_2_run_4_NS_11-41-35_AM.csv
            Loading: sub_2_run_3_SN_pt_1_11-40-17_AM.csv
            Loading: Sub_1_Run_2_SN_11-47-57_AM.csv
            Loading: Sub_1_Run_3_NS_11-49-29_AM.csv
            Loading: sub_3_run_4_NS_11-26-08_AM.csv
            Loading: Sub3_run7_SN_11-34-22_AM.csv
            '''
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
            if 'noStep' in fileData.columns:
                noStep = fileData['noStep'].values
            else:
                print(f"Warning: 'noStep' column missing in {fileName}. Filling with zeros.")
                noStep = np.zeros(len(fileData))
            sTime_str = fileData['Time'].values
            # Convert to tensors
            h_left_tensor = torch.tensor(h_left, dtype=torch.float32)
            h_right_tensor = torch.tensor(h_right, dtype=torch.float32)
            t_left_tensor = torch.tensor(t_left, dtype=torch.float32)
            t_right_tensor = torch.tensor(t_right, dtype=torch.float32)


            # Sliding window from the data
            self.sldWin(h_left_tensor, noStep, 1, sTime_str)
            self.sldWin(h_right_tensor, noStep, 1, sTime_str)
            self.sldWin(t_left_tensor, noStep, 2, sTime_str)
            self.sldWin(t_right_tensor, noStep, 2, sTime_str)
        # Done File

    def sldWin(self, data, noStep, label, sTime):
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            self.sTime.append(sTime[i])
            window = data[i:i + self.window_size]
            noStep_win = noStep[i:i + self.window_size]

            window = self.standerdize(window) # Standardize each block to it's self

            self.samples.append(window.unsqueeze(-1))  # shape: (window_size, 1)
            noStep_sum = np.sum(noStep_win)
            print(f"frame: {i}:{i+self.window_size}, noStepSum: {noStep_sum}, Label: {label}")
            if noStep_sum > 10: 
                self.labels.append(0) 
            else:
                self.labels.append(label) 

    #def normalize(self, dataBlock):

    def standerdize(self, dataBlock):
        mean = torch.mean(dataBlock)
        std = torch.std(dataBlock)
        return (dataBlock - mean)/std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.sTime[idx]  # shape: (window_size, 2)
    
### Model ###
class nNet(nn.Module ):
    def __init__(self, input_size, nClasses):
        super(nNet, self).__init__()
        layerSize = 64
        self.fc1 = nn.Linear(input_size, layerSize)
        self.fc2 = nn.Linear(layerSize, 128)  
        self.fc3 = nn.Linear(128, layerSize)  
        self.classifyer = nn.Linear(layerSize, nClasses)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (batch, window_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.classifyer(x)
        return x

## Plotting ###
def plot_data(i, window, label, sTime):
    t = np.linspace(0, windowLen_s, num=len(window), endpoint=False)
    title_str = f"Frame {i}: Start Time: {sTime} {window.shape}, label: {label}"
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
    for i, (window, label, sTime) in enumerate(dataset):
        plot_data(i*sampleFreq_hz, window, label, sTime)
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
    for batch_window, batch_label, _ in train_loader:

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
    for batch_window, batch_label, _ in test_loader:
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
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
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
# Optimizer -- Hill analogy