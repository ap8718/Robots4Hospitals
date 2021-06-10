import cv2 as cv
import numpy as np
import mediapipe as mp
import torch

def inbox_cnt(xmin, ymin, xmax, ymax, x, y):
  if (xmin < x < xmax) and (ymin < y < ymax):
    return True
  else:
    return False

# threshold from static images:
def get_rej_threshold(INPUT_IMAGE_PATH='', OUTPUT_IMAGE_PATH=''):
  max_area = max_z = 0
  mp_hands = mp.solutions.hands
  mp_drawing = mp.solutions.drawing_utils

  with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8) as static_hands:
    image = cv.flip(cv.imread(INPUT_IMAGE_PATH), 1)
    results = static_hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      x_min = y_min = 1e6
      x_max = y_max = 0
      for n in range (21):
          x, y, z = hand_landmarks.landmark[n].x, hand_landmarks.landmark[n].y, hand_landmarks.landmark[n].z
          x_t, y_t = x * image_width, y * image_height
          x_min, y_min, x_max, y_max = min(x, x_min), min(y, y_min), max(x, x_max), max(y, y_max)
          max_z = max(max_z, abs(z))
      area = (y_max - y_min) * (x_max - x_min)
      max_area = max(area, max_area)
      mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv.imwrite(OUTPUT_IMAGE_PATH, cv.flip(annotated_image, 1))
    return max_area, max_z

#video analysis:
def visor_doff(threshold, threshold_z, reject_ratio, lamda, INPUT_MP4_PATH='', OUTPUT_MP4_PATH='',MODEL_PATH=''):
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)
  mp_drawing = mp.solutions.drawing_utils
  model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

  # cap = cv.VideoCapture(INPUT_MP4_PATH) #put video directory here
  cap = cv.VideoCapture(0)
  fourcc = cv.VideoWriter_fourcc(*'MP4V')
  fps = cap.get(cv.CAP_PROP_FPS)
  fcount  = cap.get(cv.CAP_PROP_FRAME_COUNT)
  videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
  out = cv.VideoWriter(OUTPUT_MP4_PATH, fourcc, fps, (videoWidth,videoHeight))

  max_cnt = max_score = 0
  xmin = ymin = xmax = ymax = 0
  cnt_list = area_list = []
  flag = False
  index = 0
  h_frame_continuous = h_frame_overall = 0
  max_h_frame_continuous = max_h_frame_overall = 0

  blank_image = np.zeros((videoHeight, videoWidth, 3), np.uint8)
  text1 = 'Rej thres_area : ' + str(round(threshold, 2))
  text2 = 'Rej thres_z-value ' + str(round(threshold_z, 2))
  cv.putText(blank_image, text1, (0, videoHeight//2), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1, cv.LINE_AA)
  cv.putText(blank_image, text2, (0, videoHeight//2 + 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1, cv.LINE_AA)
  for i in range (int(3*fps)):
    out.write(blank_image)

  while cap.isOpened():
      cnt = score = 0
      ret, frame = cap.read()
      prev_max_score = max_score
      prev_max_cnt = max_cnt

      if not ret:
        print("EOF. Exited")
        break
      threshold_text = 'Rej thresholds: ' + str(round(threshold, 2)) + ' / ' + str(round(threshold_z, 2))
      image = cv.flip(frame, 1)
      visor = model(cv.cvtColor(image, cv.COLOR_BGR2RGB), size=640)
      box_list = visor.xyxy[0].tolist()
      if box_list:
        xmin, ymin, xmax, ymax = box_list[0][:4]
      else:
        xmin = ymin = xmax = ymax = 0
        cv.putText(frame, 'No visor detected',(10,40), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1, cv.LINE_AA)
        out.write(frame)
        continue

      results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
      max_text1 = 'Max contact: ' + str(max_cnt)
      max_text2 = 'Max obj score: ' + str(round(max_score, 2))

      if not results.multi_handedness or not results.multi_hand_landmarks:
        image = cv.flip(image, 1)
        results2 = model(image, size=640)
        results2.render()
        cv.putText(results2.imgs[0], 'No hand detected',(10,40), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1, cv.LINE_AA)
        cv.putText(results2.imgs[0], max_text1, (10,70), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1, cv.LINE_AA)
        cv.putText(results2.imgs[0], max_text2, (10,100), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1, cv.LINE_AA)
        out.write(results2.imgs[0])
        continue

      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        x_min = y_min = 1e6
        x_max = y_max = 0
        prev_cnt, prev_score = cnt, score
        tmp = score
        z_list = []
        for n in range (21):
          x, y, z = hand_landmarks.landmark[n].x, hand_landmarks.landmark[n].y, hand_landmarks.landmark[n].z
          x_t, y_t = x * image_width, y * image_height
          z_list.append(z)
          if inbox_cnt(xmin, ymin, xmax, ymax, x_t, y_t):
            cnt += 1
            if (z > 0) and (z < reject_ratio * threshold_z):
              score += 1
            elif (z < 0) and (z < reject_ratio * threshold_z):
              score += 0.3 #parameter to tune
          x_min, y_min, x_max, y_max = min(x, x_min), min(y, y_min), max(x, x_max), max(y, y_max)

        area = (y_max - y_min) * (x_max - x_min)
        z_mean = abs(sum(z_list) / len(z_list))
        z_list.clear()
        if index == 0:
          threshold = area
          threshold_z = z_mean
        if (area > reject_ratio * threshold) or (z_mean > reject_ratio * threshold_z): #parameter to tune
          cnt, score = prev_cnt, prev_score
        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      index += 1
      annotated_image = cv.flip(annotated_image, 1)
      results2 = model(annotated_image, size=640)
      box_list_2 = results2.xyxy[0].tolist()
      results2.render()
      max_cnt, max_score = max(max_cnt, cnt), max(max_score, score)

      if box_list and (not box_list_2): # trick to avoid mediapipe bugs
        max_cnt = prev_max_cnt
        max_score = prev_max_score
        cnt = 0
        score = 0
        cv.putText(results2.imgs[0], "Rejected", (10,160), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)

      text1 = 'Current contact: ' + str(cnt)
      text2 = 'Objective score: ' + str(round(abs(score), 2))
      max_text1 = 'Max contact: ' + str(max_cnt)
      max_text2 = 'Max obj score: ' + str(round(max_score, 2))
      cv.putText(results2.imgs[0], text1, (10,40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
      cv.putText(results2.imgs[0], text2, (10,70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
      cv.putText(results2.imgs[0], max_text1, (10,100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
      cv.putText(results2.imgs[0], max_text2, (10,130), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
      out.write(results2.imgs[0])

      cv.imshow("Output", results2.imgs[0])

      #debug stuff
      #if (cnt > 10):
      #  cv2_imshow(results2.imgs[0])

      if (prev_max_score > lamda) and (max_score > lamda):
        max_h_frame_continuous += 1
        max_h_frame_continuous = max(max_h_frame_continuous, h_frame_continuous)

      if (max_score > lamda):
        h_frame_overall += 1
        max_h_frame_overall = max(max_h_frame_overall, h_frame_overall)

      if cv.waitKey(2) == ord('q'):
          break

  blank_image = np.zeros((videoHeight, videoWidth, 3), np.uint8)
  text1 = 'Max obj score: ' + str(round(max_score, 2))
  cls = 'Visor touched' if max_score > lamda else 'Visor not touched'
  text2 = 'Result: ' + cls
  cv.putText(blank_image, text1, (0, videoHeight//2), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1, cv.LINE_AA)
  cv.putText(blank_image, text2, (0, videoHeight//2 + 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1, cv.LINE_AA)
  for i in range (int(3*fps)):
    out.write(blank_image)

  cap.release()
  out.release()
  cv.destroyAllWindows()

  return max_score, max_h_frame_continuous, max_h_frame_overall
