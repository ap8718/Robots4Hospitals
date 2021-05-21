import torch
import tensorflow as tf
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

def detect_and_predict_mask(img, faceNet, maskNet):
	# grab the dimensions of the img and then construct a blob from it
	(h, w) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.55:  # args["confidence"]
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the img
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = img[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)


# VISOR DETECTOR
#######################################################################################################

visorModel = torch.hub.load('ultralytics/yolov5', 'custom', path='visor.pt')  # custom model
# Image

resultlist = []
for i in range(0,5):
    img =cv2.imread(r"imagesFromPepper/analysis" + str(i) + ".png")
    result = visorModel(img)
    try:
        result = int(result.xyxy[0][0][5])
    except:
        result = -1
    resultlist.append(result)

print(resultlist)
mode = max(set(resultlist), key=resultlist.count)
print(mode)

f = open('VisorText','w')

if mode == 1:
        f.write('Visor not detected')
elif mode == 0 :
    f.write('Visor Detected')
else:
    f.write('Visor not detected')




# MASK DETECTOR
#######################################################################################################


# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--face", type=str,
# 	default="face_detector",
# 	help="path to face detector model directory")
# ap.add_argument("-m", "--model", type=str,
# 	default="mask_detector.h5",
# 	help="path to trained face mask detector model")
# args = vars(ap.parse_args())

# print("[INFO] loading face detector model...")
# prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
# weightsPath = os.path.sep.join([args["face"],
# 	"res10_300x300_ssd_iter_140000.caffemodel"])
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# # load the face mask detector model from disk
# print("[INFO] loading face mask detector model...")
# maskNet = load_model('mask_detector.h5')

# # initialize the video stream and allow the camera sensor to warm u
# 	# grab the img from the threaded video stream and resize it to have a maximum width of 400 pixels
# img = cv2.imread(r"imagesFromPepper/analysis0.png") 

# # detect faces in the img and determine if they are wearing a face mask or not
# (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)

# for (box, pred) in zip(locs, preds):
#     # unpack the bounding box and predictions
#     (startX, startY, endX, endY) = box
#     (incorrectMask, mask, withoutMask) = pred

#     f = open('MaskText','w')

#     # determine the class label and color we'll use to draw the bounding box and text
#     if(mask > withoutMask) and (mask > incorrectMask):
#         label = "Mask"
#         f.write('Mask correctly detected')
        
    
#         color = (0, 255, 0)
    
#     elif(withoutMask > mask) and (withoutMask > incorrectMask):
#         label = "No mask"
#         color = (0, 0, 255)
#         f.write('No mask detected')

#     else:
#         label = "Incorrectly Worn mask"
#         color = (0, 255, 255)
#         f.write('Mask worn incorrectly detected')

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

    if (num_hands == 1 and num_filt_hands == 1) :
        result = 'single glove detected '
    if (num_hands == 1 and num_filt_hands == 0) :
        result = 'single hand detected'
    if (num_hands == 2 and num_filt_hands == 2) :
        result = 'both gloves detected'
    if (num_hands == 2 and num_filt_hands == 1) :
        result = 'a glove and hand detected'
    if (num_hands == 2 and num_filt_hands == 0) :
        result = 'only hands detected'
    # else:
    #     result = 'none'

    f = open("GloveText", 'w')
    f.write(result)


    return result
resultlist = []
for i in range(0,5):
    img =cv2.imread(r"imagesFromPepper/analysis" + str(i) + ".png")
    height, width, channels = img.shape     
    result = detect_gloves(img, showImg = True)
    resultlist.append(result)

mode = max(set(resultlist), key=resultlist.count)


f = open('GloveText','w')

f.write(mode)

#### GOWN DETECTOR

gownModel = torch.hub.load('ultralytics/yolov5', 'custom', path='gown_harsh.pt') 
# Image

# Inference


resultlist = []
for i in range(0,5):
    img =cv2.imread(r"imagesFromPepper/analysis" + str(i) + ".png")
    result = gownModel(img)
    result.print()
    try:
        result = int(result.xyxy[0][0][5])
        print(result)
    except:
        result = -1
    resultlist.append(result)

mode = max(set(resultlist), key=resultlist.count)

f = open('GownText','w')

if mode == 1:
        f.write('Gown not detected')
elif mode == 0 :
    f.write('Gown Detected')
else:
    f.write('Gown not detected')


