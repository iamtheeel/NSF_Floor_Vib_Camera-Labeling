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

train_data = datasets.Flowers102(
    root="Flowers102", #MNIST, # where to download data to?
    download=True, # download data if it doesn't exist on disk
)


print(f"Data: {len(train_data._image_files)}, labels: {len(train_data._labels)}")


itemno = 20
filetype = False

for i, item in enumerate(train_data._image_files):
    image, label = train_data[i+itemno]
    image = np.array(image)  # shape: (H, W, 3), RGB
    if filetype == False:
        print(f"Data type: {type(image)}") 
        filetype = True
    print(f"Shape: {image.shape}")
    rgbbgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Item no. {i+itemno}, label: {train_data.classes[label]}")
    cv2.imshow("label", rgbbgr)
    key = cv2.waitKey(int(0))
    if key == ord('q') & 0xFF: exit()



#image, label = train_data[5000]
#image = np.array(image)  # shape: (H, W, 3), RGB
#cv2.imshow("label", image)
#cv2.waitKey(0)

# Loop over the data