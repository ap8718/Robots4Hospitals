import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'custom', path='visor.pt')  # custom model
# Image
img = 'https://ultralytics.com/images/zidane.jpg'
# Inference
results = model(img)
results.print()
results.save()
