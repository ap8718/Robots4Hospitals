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
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    num_hands_detected = 0

    if(detections['detection_scores'][0] > 0.7):
        num_hands_detected += 1
        ymin1, xmin1, ymax1, xmax1 = tuple(detections['detection_boxes'][0].tolist())
        ymin1, xmin1, ymax1, xmax1 = int(height*ymin1), int(width*xmin1), int(height*ymax1), int(width*xmax1)
        print(ymin1, xmin1, ymax1, xmax1)
        hand1 = image_np[ymin1:ymax1, xmin1:xmax1]
        # cv2.imshow('hand 1 detection', hand1)

        hand1 = hand1.reshape((-1,3))
        if(hand1[ np.logical_and(hand1[:,0] > 75, hand1[:,0] < 125), 0 ].shape[0] >= 3000):
            detectedClass1 = "Glove"
        else:
            detectedClass1 = "Hand"
        print("{} detected!".format(detectedClass1))
        cv2.putText(image_np_with_detections, detectedClass1, (10,450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 225, 0), cv2.LINE_AA)


    if(detections['detection_scores'][1] > 0.7):
        num_hands_detected += 1
        ymin2, xmin2, ymax2, xmax2 = tuple(detections['detection_boxes'][1].tolist())
        ymin2, xmin2, ymax2, xmax2 = int(height*ymin2), int(width*xmin2), int(height*ymax2), int(width*xmax2)
        print(ymin2, xmin2, ymax2, xmax2)
        hand2 = image_np[ymin2:ymax2, xmin2:xmax2]
        # cv2.imshow('hand 2 detection', hand2)

        hand2 = hand2.reshape((-1,3))
        if(hand2[ np.logical_and(hand2[:,0] > 75, hand2[:,0] < 125), 0 ].shape[0] >= 3000):
            detectedClass2 = "Glove"
        else:
            detectedClass2 = "Hand"
        print("{} detected!".format(detectedClass2))
        cv2.putText(image_np_with_detections, detectedClass2, (10,550), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 225, 0), cv2.LINE_AA)

    print("{} hands detected".format(num_hands_detected))

    # print("Showing image...")
    cv2.imshow('object detection',  image_np_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
