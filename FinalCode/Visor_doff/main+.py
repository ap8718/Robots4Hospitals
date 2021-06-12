import cv2 as cv
import numpy as np
import mediapipe as mp
import torch
import func2 as f


#take a static image with hands very close to visor but without contact
threshold, threshold_z, area_bbox = f.get_rej_threshold(INPUT_IMAGE_PATH='/content/m1.png',
OUTPUT_IMAGE_PATH='/content/output.jpg',
MODEL_PATH='/content/Robots4Hospitals/FinalCode/Visor_doff/visortrack++.pt')

#define some parameters:
#reject_ratio: range from 1 to infinity, the smaller the more
#accurate in rejecting the 'hands in front of visor but not touching' cases.
reject_ratio = 1.05

#lamda: a classifcation boundary. for example, in this case if
#max(obj_score > 10) the result will be classfied as visor touched.
lamda = 5

p1, p2, p3 = f.visor_doff(threshold, threshold_z, area_bbox, reject_ratio, lamda,
INPUT_MP4_PATH='/content/visor_hands_input2.avi',
OUTPUT_MP4_PATH='/content/output.mp4',
MODEL_PATH='/content/Robots4Hospitals/FinalCode/Visor_doff/visortrack++.pt')

print("Maximum obj score achieved: {0}".format(p1))
print("Number of continuous frames with obj score > lamda: {0}".format(p2))
print("Number of overall frames with obj score > lamda : {0}".format(p3))
