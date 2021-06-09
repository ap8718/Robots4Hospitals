import torch
from torchvision.utils import save_image
import cv2
import numpy
model = torch.hub.load('ultralytics/yolov5', 'custom', path='visor.pt')  # custom model
# Image
img = cv2.imread(r"imagesFromPepper/camImage.png")   
# Inference
results = model(img)
results.print()
try:
    result = int(results.xyxy[0][0][5])
except:
    result = 9
f = open('VisorText','w')

if result == 1:
        f.write('Visor not detected')
elif result == 0 :
    f.write('Visor Detected')
else:
    f.write('Error please scan again')

print(result)
