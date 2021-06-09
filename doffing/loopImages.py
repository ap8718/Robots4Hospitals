# import cv2
from PIL import Image
import numpy as np
import os

done = False
counter = 0

while True:
    try:
        im = Image.open("recordings/image" + str(counter) + ".png")
    except IOError:
        break

print("Num of frames: " + str(counter))