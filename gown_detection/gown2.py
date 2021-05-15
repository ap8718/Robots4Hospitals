import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'custom', path='gown_harsh.pt') 
# Image
img = cv2.imread(r"imagesFromPepper/camImage.png")
# Inference
results = model(img)
results.print()
results.save()
