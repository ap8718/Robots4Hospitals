WORKSPACE_PATH = 'Tensorflow/workspace'
MODEL_PATH = WORKSPACE_PATH+'/models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np

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

def detect_gloves(frame, showImg = False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    filter_value = 10
    lower_blue = np.array([110,filter_value,filter_value])
    # upper_blue = np.array([130,255,255])
    upper_blue = np.array([150,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    filtered_frame = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    #cv2.imshow('res',res)

    filtered_image = np.array(filtered_frame)

    if showImg:
        cv2.imshow('filtered_image',  cv2.resize(filtered_image, (800, 600)))

    image_np = np.array(frame)
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

    if showImg:
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=2,
                    min_score_thresh=.6,
                    agnostic_mode=False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 50)
        if (num_hands == 1 and num_filt_hands == 1) :
            cv2.putText(image_np_with_detections, 'glove', org, font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        if (num_hands == 1 and num_filt_hands == 0) :
            cv2.putText(image_np_with_detections, 'hand', org, font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        if (num_hands == 2 and num_filt_hands == 2) :
            cv2.putText(image_np_with_detections, 'gloves', org, font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        if (num_hands == 2 and num_filt_hands == 1) :
            cv2.putText(image_np_with_detections, 'glove + hand', org, font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        if (num_hands == 2 and num_filt_hands == 0) :
            cv2.putText(image_np_with_detections, 'hands', org, font, 2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        cv2.imwrite('glovedetected.png', image_np_with_detections)

    return result



if __name__ == "__main__":
    frame = cv2.imread(r"imagesFromPepper/camImage.png")   
    height, width, channels = frame.shape     
    print(height, width)
    result = detect_gloves(frame, showImg = True)
    print(result)
