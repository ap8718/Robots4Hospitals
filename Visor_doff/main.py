import func

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

model = torch.hub.load('ultralytics/yolov5', 'custom', path='visor_track+.pt')

#take a static image with hands very close to visor but no contact
threshold, threshold_z = get_rej_threshold(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)

#define some parameters:

reject_ratio = 1.5
lamda = 10
p1, p2, p3 = visor_doff(INPUT_MP4_PATH, OUTPUT_MP4_PATH, threshold, threshold_z, reject_ratio, lamda)
