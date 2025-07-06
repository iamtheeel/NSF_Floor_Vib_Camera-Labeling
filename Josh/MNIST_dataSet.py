####
#   Joshua Mehlman
#   STARS Summer 2025
#   Dr J Lab
###
# Playing with pytorch
# Load the MNIST Dataset
####

from torchvision import datasets
from torchvision.transforms import ToTensor

import cv2  # pip install opencv-python
import numpy as np

train_data = datasets.MNIST(
    root="MNIST", #MNIST, # where to download data to?
    download=True, # download data if it doesn't exist on disk
)


print(f"Data: {len(train_data.data)}, labels: {len(train_data.targets)}")

image, label = train_data[50]
image = np.array(image)  # shape: (H, W, 3), RGB
cv2.imshow("label", image)
cv2.waitKey(0)

# Loop over the data