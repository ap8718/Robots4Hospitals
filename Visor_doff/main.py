import cv2 as cv
import numpy as np
import mediapipe as mp
import torch
import func

#take a static image with hands very close to visor but without contact
threshold, threshold_z = func.get_rej_threshold(INPUT_IMAGE_PATH='example.jpg', 
OUTPUT_IMAGE_PATH='example_output.jpg') 

#define some parameters:
#reject_ratio: range from 1 to infinity, the smaller the more 
#accurate in rejecting the 'hands in front of visor but not touching' cases.
reject_ratio = 1.5

#lamda: a classifcation boundary. for example, in this case if
#max(obj_score > 10) the result will be classfied as visor touched.
lamda = 10 

p1, p2, p3 = func.visor_doff(threshold, threshold_z, reject_ratio, lamda,
INPUT_MP4_PATH='example.mp4',
OUTPUT_MP4_PATH='example_output.mp4',
MODEL_PATH='visor_track+.pt')

print("Maximum obj score achieved: {0}".format(p1))
print("Number of continuous frames with obj score > lamda: {0}".format(p2)) 
print("Number of overall frames with obj score > lamda : {0}".format(p3)) 

