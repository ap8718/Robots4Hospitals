# from PIL import Image
import cv2
import numpy as np
import os
import time
# import compiledmodels2 as cm2

prev_img = np.zeros((60,60))
img = np.zeros((60,60))

read = False

print("Running loop...")
while True:
    try:
        # img = np.array(Image.open(r"analysis0.png"))
        img = cv2.imread(r"imagesFromPepper/analysis0.png")
        if read and not (np.all(img == prev_img)):
            print("Image recieved!")
            os.system("python3 CompiledModels.py")
            # cm2.main()
            # time.sleep(30)
            read = False
        else:
            read = True
        prev_img = img

    except KeyboardInterrupt:
        print('\nInterrupted')
        break
    except:
        continue
