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

train_data = datasets.EMNIST(
    root="EMNIST", #MNIST, # where to download data to?
    split="letters", # "letters" for EMNIST Letters, "digits" for EMNIST Digits
    download=True, # download data if it doesn't exist on disk
)

print("Hey we're running this code")
print(f"Data: {len(train_data.data)}, labels: {len(train_data.targets)}")

#image, label = train_data[50]
#image = np.array(image)  # shape: (H, W, 3), RGB
#cv2.imshow("label", image)
#key1 = cv2.waitKey(0)
#if key1 & 0xFF == ord('q') : exit()
#cv2.destroyAllWindows()

# Loop over the data
for i in range(10):
    print("in loop")
    image, label = train_data[i]
    image = np.array(image)  # shape: (H, W, 3), RGB
    cv2.imshow("label2", image)
    print(f"Label: {label}")
    key2 = cv2.waitKey(0)
    if key2 == ord('q') & 0xFF: exit()

