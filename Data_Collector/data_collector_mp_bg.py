import cv2
import mediapipe as mp
import math as m
import time
from utils import BOUNDING_SIDE
import os

bbox = BOUNDING_SIDE.bbox()
folder = 'Dataset/4'
count = 0

if not os.path.exists(folder):
    os.makedirs(folder)
    
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

offset = 40
imgSize = 300

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                bbox.bbox_draw(img=image, hand_landmarks=hand_landmarks)
                bbox.bbox_show(img=image)

                x, y, w, h = bbox.bbox_coord(img=image, hand_landmarks=hand_landmarks)

        cv2.imshow("Image", image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            count += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img=image)
            print(count)
        elif key == 27:
            exit()

cap.release()
cv2.destroyAllWindows()