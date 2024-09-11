#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import mediapipe as mp
import graphical_utils as gu
import hand_gesture_utils as hgu
import utils

class handDetector():
    def __init__(self, mode=False, model_complexity=1, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 16:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 20:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    ''' Test della rilevazione della mano utilizzando la classe handDetector e la classe Camera. '''

    # Inizializzazione del nodo ROS
    rospy.init_node('hand_tracking_test_node')

    # Parametri della camera
    cam_name = 'Kinect'  # Modificato per usare la webcam

    # Caricamento dei parametri di calibrazione della camera (se richiesti)
    path = '/home/jacopo/URProject/src/hands-free-project/src/yaml/'
    camera_calibration = utils.yaml2dict(path + 'camera_calibration.yaml')
    K = camera_calibration['K']
    D = camera_calibration['D']
    R = camera_calibration['R']
    t = camera_calibration['t']

    # Inizializzazione della camera
    if cam_name == 'Kinect':
        camera = utils.Kinect(enable_rgb=True, enable_depth=False, need_bigdepth=False, need_color_depth_map=False, K=K, D=D)
    else:
        camera = utils.Camera(enable_rgb=True, camera_id=0)

    # Inizializzazione dell'oggetto handDetector
    detector = handDetector(detectionCon=0.7)  # Configurazione opzionale dei parametri

    while not rospy.is_shutdown():
        # Acquisizione del frame dalla camera
        camera.acquire(False)
        frame = camera.RGBundistorted.copy()

        # Rilevazione delle mani usando il detector
        frame = detector.findHands(frame)

        # Trova la posizione della mano (o delle mani) nel frame
        lmList = detector.findPosition(frame)
        
        # Verifica e visualizzazione dei punti rilevati
        if lmList:
            rospy.loginfo('Punti della mano: ' + str(lmList))
            # Puoi aggiungere ulteriori funzioni di visualizzazione o elaborazione qui

        # Mostra il frame con i risultati
        cv2.imshow('Hand Detection', frame)

        # Esci dal loop se si preme 'q'
        if cv2.waitKey(25) == ord('q'):
            break

    # Ferma la camera e chiudi tutte le finestre
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()