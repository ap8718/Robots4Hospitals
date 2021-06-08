import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# mp_holistic = mp.solutions.holistic

def analyseGownDoffing(image, results):
    h, w, c = image.shape

    danger = False

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

        if LhandWithinShoulders.any() or RhandWithinShoulders.any():
            # handColour = (0,127,255)
            if sum(LhandAboveNeck) >= 3 or sum(RhandAboveNeck) >= 3:    # 2 or more points
                danger = True

        if(right_dist_elbow < 0.37 or left_dist_elbow < 0.37):
            danger = True

        if danger:
            handColour = (0,0,255)

        # Colours in BGR
        cv2.circle(image, (cx15, cy15), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx17, cy17), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx19, cy19), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx21, cy21), 5, handColour, cv2.FILLED)

        # cv2.line(image, (cx19, cy19), (cx7, cy7), (0, right_dist_shoulder, 255-right_dist_shoulder), 3)

        cv2.circle(image, (cx16, cy16), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx18, cy18), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx20, cy20), 5, handColour, cv2.FILLED)
        cv2.circle(image, (cx22, cy22), 5, handColour, cv2.FILLED)

        cv2.circle(image, (cx12, cy12_new), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (cx11, cy11_new), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (cx_neck, cy_neck), 5, (0, 0, 255), cv2.FILLED)

        cv2.line(image, (cx11, cy11_new), (cx12, cy12_new), (0, 255, 0), 1)

        if danger:
            # print("DANGER")
            cv2.putText(image, "DANGER", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3, cv2.LINE_AA)

    return danger

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # image = cv2.resize(image, dsize=(480, 320), interpolation = cv2.INTER_CUBIC)
            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose Detections
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            danger = analyseGownDoffing(image, results)
            if danger:
                print("DANGER")

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
