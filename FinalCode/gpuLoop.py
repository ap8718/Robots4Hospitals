import cv2
import numpy as np
import os
import time

prev_img = np.zeros((60,60))
img = np.zeros((60,60))

read = False

print("Running loop...")
while True:
    try:
        img = cv2.imread(r"imagesFromPepper/analysis0.png")
        if read and not (np.all(img == prev_img)):
            print("Image recieved!")
            os.system("python3 CompiledModels.py")
            read = False
        else:
            read = True
        prev_img = img

    except KeyboardInterrupt:
        print('\nInterrupted')
        break
    except:
        continue
