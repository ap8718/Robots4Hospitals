import cv2
import numpy as np
import os
import time

donning_prev_img = np.zeros((480,320))
donning_img = np.zeros((480,320))

gown_prev_img = np.zeros((480,320))
gown_img = np.zeros((480,320))

visor_prev_img = np.zeros((480,320))
visor_img = np.zeros((480,320))

gloves_prev_img = np.zeros((480,320))
gloves_img = np.zeros((480,320))

donning_read = False
gown_read = False
visor_read = False
gloves_read = False

print("Running loop...")
while True:
    try:
        donning_img = cv2.imread(r"imagesFromPepper/analysis0.png")
        
        cap = cv2.VideoCapture(r"Gown_doff/gown.avi")
        _, gown_img = cap.read()

        cap = cv2.VideoCapture(r"Visor_doff/visor.avi")
        _, visor_img = cap.read()

        cap = cv2.VideoCapture(r"Glove_doff/gloves.avi")
        _, gloves_img = cap.read()

        if donning_read and not (np.all(donning_img == donning_prev_img)):
            print("Donning recieved!")
#            os.system("python3 CompiledModels.py")
            donning_read = False
        else:
            donning_read = True

        if gown_read and not (np.all(gown_img == gown_prev_img)):
            print("Gown recieved!")
            os.chdir("Gown_doff")
#            os.system("python3 gownDoffing.py")
            os.chdir("..")
            gown_read = False
        else:
            gown_read = True

        if visor_read and not (np.all(visor_img == visor_prev_img)):
            print("Visor recieved!")
            os.chdir("Visor_doff")
#            os.system("python3 visorDoffing.py")
            os.chdir("..")
            visor_read = False
        else:
            visor_read = True

        if gloves_read and not (np.all(gloves_img == gloves_prev_img)):
            print("Gloves recieved!")
            os.chdir("Glove_doff")
#            os.system("python3 gloveDoffing.py")
            os.chdir("..")
            gloves_read = False
        else:
            gloves_read = True

#        else:
#            donning_read = True
#            gown_read = True
#            visor_read = True
#            gloves_read = True

        donning_prev_img = donning_img
        gown_prev_img = gown_img
        visor_prev_img = visor_img
        gloves_prev_img = gloves_img

    except KeyboardInterrupt:
        print('\nInterrupted')
        break
    except:
        continue


