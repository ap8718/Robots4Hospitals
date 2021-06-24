import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def inference(box_list):
  gown = incomplete_gown = False
  incomplete_gown_list = []
  for item in box_list:
    if item[5] == 0:
      gown = True
    elif item[5] == 1:
      incomplete_gown = True
      xmin, ymin, xmax, ymax = item[:4]
      l = []
      l.extend((xmin, ymin, xmax, ymax))
      incomplete_gown_list.append(l)
  return gown, incomplete_gown, incomplete_gown_list

def analyseGownDoffing(image, results, incomplete_gown, incomplete_gown_list):
    h, w, c = image.shape
    danger = False
    danger_text = ''
    points = np.array([])
    if(results.pose_landmarks):
        points = np.array([(data_point.x, data_point.y, data_point.z) for data_point in results.pose_landmarks.landmark])

        #Left hand
        cx15, cy15, cz15 = int(w*points[15][0]), int(h*points[15][1]), int(h*points[15][2])     #Left wrist
        cx17, cy17, cz17 = int(w*points[17][0]), int(h*points[17][1]), int(h*points[17][2])     #Left pinky
        cx19, cy19, cz19 = int(w*points[19][0]), int(h*points[19][1]), int(h*points[19][2])     #Left index
        cx21, cy21, cz21 = int(w*points[21][0]), int(h*points[21][1]), int(h*points[21][2])     #Left thumb

        #Left hand
        cx16, cy16, cz16 = int(w*points[16][0]), int(h*points[16][1]), int(h*points[16][2])     #Right wrist
        cx18, cy18, cz18 = int(w*points[18][0]), int(h*points[18][1]), int(h*points[18][2])     #Right pinky
        cx20, cy20, cz20 = int(w*points[20][0]), int(h*points[20][1]), int(h*points[20][2])     #Right index
        cx22, cy22, cz22 = int(w*points[22][0]), int(h*points[22][1]), int(h*points[22][2])     #Right thumb

        cx11, cy11 = int(w*points[11][0]), int(h*points[11][1])     #Left shoulder
        cx12, cy12 = int(w*points[12][0]), int(h*points[12][1])     #Right shoulder

        cx13, cy13 = int(w*points[13][0]), int(h*points[13][1])     #Left elbow
        cx14, cy14 = int(w*points[14][0]), int(h*points[14][1])     #Right elbow

        cx23, cy23 = int(w*points[23][0]), int(h*points[23][1])     #Left hip
        cx24, cy24 = int(w*points[24][0]), int(h*points[24][1])     #Right hip

        height = int(abs(cy12 - cy24))

        right_dist_elbow = int(abs(cx15 - cx14) + abs(cy15 - cy14))             #Manhattan distance
        left_dist_elbow = int(abs(cx16 - cx13) + abs(cy16 - cy13))

        right_dist_elbow /= height
        left_dist_elbow /= height


        right_dist_shoulder = int(abs(cy12 - cy24))          #Manhattan distance
        left_dist_shoulder = int(abs(cy11 - cy23))           #Manhattan distance

        scale = 1.07    # Hip to tip of shoulder / hip to middle of shoulder

        right_dist_shoulder *= scale
        left_dist_shoulder *= scale

        cy12_new = int(cy24-right_dist_shoulder)
        cy11_new = int(cy23-left_dist_shoulder)

        #Neck
        cx_neck, cy_neck = int((cx12 + cx11)/2), int((cy12_new + cy11_new)/2)

        LHands = np.array([
                [cx15, cy15],
                [cx17, cy17],
                [cx19, cy19],
                [cx21, cy21],
        ])
        RHands = np.array([
                [cx16, cy16],
                [cx18, cy18],
                [cx20, cy20],
                [cx22, cy22],
        ])

        handColour = (0,255,0)

        LhandWithinShoulders = np.logical_and(cx12 < LHands[:,0],  LHands[:,0] < cx11)
        RhandWithinShoulders = np.logical_and(cx12 < RHands[:,0],  RHands[:,0] < cx11)

        LhandAboveNeck = (cy_neck > LHands[:,1])
        RhandAboveNeck = (cy_neck > RHands[:,1])

        LhandOnTorso = np.logical_and(LHands[:,0] > cx12, LHands[:,0] < cx11).all() and np.logical_and(LHands[:,1] > cy12_new, LHands[:,1] < cy24).all()
        RhandOnTorso = np.logical_and(RHands[:,0] > cx12, RHands[:,0] < cx11).all() and np.logical_and(RHands[:,1] > cy12_new, RHands[:,1] < cy24).all()

        handOnTorso = False
        if LhandOnTorso or RhandOnTorso:
            handOnTorso = True

        if LhandWithinShoulders.any() or RhandWithinShoulders.any():
            # handColour = (0,127,255)
            if sum(LhandAboveNeck) >= 3 or sum(RhandAboveNeck) >= 3:    # 2 or more points
                danger = True
                danger_text = 'Neck Touch Danger'

        if(right_dist_elbow < 0.37 or left_dist_elbow < 0.37):
            danger = True
            danger_text = 'Arm Touch Danger'

        if incomplete_gown:
          for l in incomplete_gown_list:
            xmin, ymin, xmax, ymax = l
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(image, (xmin, ymax), (xmax, ymin), (255,255,255),3)
            X_LhandWithinBoxes = np.logical_and(xmin < LHands[:,0],  LHands[:,0] < xmax)
            Y_LhandWithinBoxes = np.logical_and(ymin < LHands[:,1],  LHands[:,1] < ymax)

            X_RhandWithinBoxes = np.logical_and(xmin < RHands[:,0],  RHands[:,0] < xmax)
            Y_RhandWithinBoxes = np.logical_and(ymin < RHands[:,1],  RHands[:,1] < ymax)

            if not (X_LhandWithinBoxes.all() and Y_LhandWithinBoxes.all() and X_RhandWithinBoxes.all() and Y_RhandWithinBoxes.all()) and handOnTorso:
              danger = True
              danger_text = 'Hands Misposition Danger' #this condition is quite harsh, we will see..

        if danger:
            handColour = (0,0,255)

        # Colours in BGR
        cv2.circle(image, (cx15, cy15), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx17, cy17), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx19, cy19), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx21, cy21), 5, handColour, cv2.FILLED)

        cv2.circle(image, (cx16, cy16), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx18, cy18), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx20, cy20), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx22, cy22), 5, handColour, cv2.FILLED)

        cv2.circle(image, (cx12, cy12_new), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (cx11, cy11_new), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (cx_neck, cy_neck), 5, (0, 0, 255), cv2.FILLED)

        cv2.line(image, (cx11, cy11_new), (cx12, cy12_new), (0, 255, 0), 1)
        cv2.line(image, (cx11, cy11_new), (cx11, cy23), (0, 255, 0), 1)
        cv2.line(image, (cx12, cy12_new), (cx12, cy24), (0, 255, 0), 1)
        cv2.line(image, (cx12, cy24), (cx11, cy23), (0, 255, 0), 1)

        if danger:
            cv2.putText(image, danger_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

    return danger, image

#INPUT_MP4_PATH = 'test.avi'
#uncomment for pepper avi videos
INPUT_MP4_PATH = 'test.mp4'
OUTPUT_MP4_PATH = 'output.mp4'

def main():

    MODEL_PATH = 'gown_harsh.pt'
    cap = cv2.VideoCapture(INPUT_MP4_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcount  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_MP4_PATH, fourcc, fps, (videoWidth,videoHeight))
    print("\nLoading gown model...\n")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    image_list = []
    dangerList = []
    sendDangerSignal = False

        # Initiate holistic model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor Feed
            if not ret:
              print("EOF. Exited")
              break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Uncomment to see performance with Pepper's resolution
            # image = cv2.resize(image, dsize=(480, 320), interpolation = cv2.INTER_CUBIC)

            res = model(image, size=640)
            box_list = res.xyxy[0].tolist()
            gown, incomplete_gown, incomplete_gown_list = inference(box_list)
            if (not gown) and (not incomplete_gown):
              cv2.putText(image, "No Gown Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
              out.write(image)
              continue

            # Make Pose Detections
            results = pose.process(image)
            if not results.pose_landmarks:
              cv2.putText(image, "No Pose Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
              out.write(image)
              continue

            # Recolor image back to BGR for rendering
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Pose Detections
            danger, image = analyseGownDoffing(image_bgr, results, incomplete_gown, incomplete_gown_list)
            if danger:
                print("DANGER")
                dangerList.append(1)
            if len(dangerList) == 5:
                sendDangerSignal = True

            f = open("GownDoffingText", "w")

            if sendDangerSignal:
                print("***CONTAMINATION DETECTED***")
                f.write("Contamination detected!")
            else:
                f.write("Gown safely doffed")

            out.write(image)

    cap.release()

if __name__ == "__main__":
    main()
