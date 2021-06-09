import torch
import tensorflow as tf
from PIL import Image
from torchvision.utils import save_image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

WORKSPACE_PATH = 'Tensorflow/workspace'
MODEL_PATH = WORKSPACE_PATH+'/models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'



# VISOR DETECTOR
#######################################################################################################

visorModel = torch.hub.load('ultralytics/yolov5', 'custom', path='visor.pt')  # custom model
# Image

resultlist = []
for i in range(0,1):
    img =Image.open(r"imagesFromPepper/analysis" + str(i) + ".png")
    resultIMG = visorModel(img)
    try:
        result = int(resultIMG.xyxy[0][0][5])
    except:
        result = 2
    print(result)

    resultlist.append(result)

print(resultlist)
mode = max(set(resultlist), key=resultlist.count)
print(mode)

f = open('Results/VisorText','w')

if mode == 1:
        f.write('Visor not detected')
elif mode == 0 :
    f.write('Visor Detected')
else:
    f.write('Visor not detected')




# MASK DETECTOR
#######################################################################################################

maskModel = torch.hub.load('ultralytics/yolov5', 'custom', path='mask.pt')  # custom model
# Image

resultlist = []
for i in range(0,1):
    img =Image.open(r"imagesFromPepper/analysis" + str(i) + ".png")
    resultIMG = maskModel(img)
    try:
        result = int(resultIMG.xyxy[0][0][5])
    except:
        result = 1
    print(result)

    resultlist.append(result)

print(resultlist)
mode = max(set(resultlist), key=resultlist.count)
print(mode)

f = open('Results/MaskText','w')

if mode == 1:
        f.write('No Mask detected')
elif mode == 0 :
    f.write('Incorrectly Worn Mask Detected')
else:
    f.write('Mask detected')



# GLOVE DETECTOR
#######################################################################################################

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21-DESKTOP-23ED9T2')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

def detect_gloves(img, showImg = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    filter_value = 10
    lower_blue = np.array([110,filter_value,filter_value])
    # upper_blue = np.array([130,255,255])
    upper_blue = np.array([150,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img,img, mask= mask)
    filtered_img = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    #cv2.imshow('res',res)

    filtered_image = np.array(filtered_img)

    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    scores = detections['detection_scores']
    hand_score1 = scores[0]
    hand_score2 = scores[1]

    x11 = int(detections['detection_boxes'][0][1] * width)
    y11 = int(detections['detection_boxes'][0][0] * height)
    x21 = int(detections['detection_boxes'][0][3] * width)
    y21 = int(detections['detection_boxes'][0][2] * height)

    x12 = int(detections['detection_boxes'][1][1] * width)
    y12 = int(detections['detection_boxes'][1][0] * height)
    x22 = int(detections['detection_boxes'][1][3] * width)
    y22 = int(detections['detection_boxes'][1][2] * height)

    counter1 = 0
    counter2 = 0

    num_hands = 0
    num_filt_hands = 0

    if hand_score1 > 0.6 :
        num_hands += 1
        for k in range(y21 - y11) :
            for i in range(x21 - x11) :
                pixel_intensity = int(filtered_image[y11 + k][x11 + i][0]) + int(filtered_image[y11 + k][x11 + i][1]) + int(filtered_image[y11 + k][x11 + i][2])
                if pixel_intensity > 0 :
                    counter1 += 1
        if counter1 * 10 >= (x21 - x11) * (y21 - y11) * 3 :
            num_filt_hands += 1

    if hand_score2 > 0.6 :
        num_hands += 1
        for k in range(y22 - y12) :
            for i in range(x22 - x12) :
                pixel_intensity = int(filtered_image[y12 + k][x12 + i][0]) + int(filtered_image[y12 + k][x12 + i][1]) + int(filtered_image[y12 + k][x12 + i][2])
                if pixel_intensity > 0 :
                    counter2 += 1
        if counter2 * 10 >= (x22 - x12) * (y22 - y12) * 3 :
            num_filt_hands += 1

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    result = ''

#    if (num_hands == 2 and num_filt_hands == 2):
#        result = 'Both gloves detected'
#    else:
#        result = 'The gloves are not correctly worn'

    print("\n num_hands = " + str(num_hands))
    print("num_filt_hands = " + str(num_filt_hands) + "\n")

    print(str(num_hands == 2 and num_filt_hands == 2) + "\n")

    if (num_hands == 1 and num_filt_hands == 1) :
        result = 'The gloves are not correctly worn'
    elif (num_hands == 1 and num_filt_hands == 0) :
        result = 'The gloves are not correctly worn'
    elif (num_hands == 2 and num_filt_hands == 2) :
        result = 'both gloves detected'
    elif (num_hands == 2 and num_filt_hands == 1) :
        result = 'The gloves are not correctly worn'
    elif (num_hands == 2 and num_filt_hands == 0) :
        result = 'The gloves are not correctly worn'
    else:
        result = 'The gloves are not correctly worn'

    f = open("Results/GloveText", 'w')
    f.write(result)

    return result

resultlist = []
for i in range(0,1):
    img =cv2.imread(r"imagesFromPepper/analysis" + str(i) + ".png")
    height, width, channels = img.shape
    result = detect_gloves(img, showImg = True)
#    resultlist.append(result)

#print(resultlist)
#mode = max(set(resultlist), key=resultlist.count)

print("\n" + result + "\n")
#f = open('Results/GloveText','w')
#f.write(result)


#### GOWN DETECTOR

gownModel = torch.hub.load('ultralytics/yolov5', 'custom', path='gown_new.pt')
# Image

# Inference


resultlist = []
for i in range(0,1):
    img =Image.open(r"imagesFromPepper/analysis" + str(i) + ".png")
    resultIMG = gownModel(img)
    try:
        result = int(resultIMG.xyxy[0][0][5])
    except:
        result = 2
    print(result)
    resultlist.append(result)

print(resultlist)
mode = max(set(resultlist), key=resultlist.count)
print(mode)

f = open('Results/GownText','w')

if mode == 1:
        f.write('Gown not detected')
elif mode == 0 :
    f.write('Gown Detected')
else:
    f.write('Gown not detected')
