#! /usr/bin/env python

import rospy
import cv2
import numpy as np
import graphical_utils as gu
import hand_gesture_utils as hgu
import utils

def main():
    ''' Test della rilevazione della mano utilizzando la classe Hand e la classe Camera. '''

    # Inizializzazione del nodo ROS
    rospy.init_node('hand_tracking_test_node')

    # Parametri della camera
    cam_name = 'Kinect'  # Modifica se necessario

    # Caricamento dei parametri di calibrazione della camera
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
        camera = utils.Camera(enable_rgb=True)

    # Inizializzazione dell'oggetto Hand
    hand = hgu.Hand(net_params=[2], threshold=0.2, debug=True)

    while not rospy.is_shutdown():
        # Acquisizione del frame dalla camera
        camera.acquire(False)
        frame = camera.RGBundistorted.copy()

        # Rilevazione dei punti della mano
        hand.mediapipe_inference(frame)

        # Verifica e visualizzazione dei punti rilevati
        if hand.points[0] is not None:
            rospy.loginfo('Punti della mano: ' + str(hand.points))
            gu.draw_gesture_info(frame, hand.inference_time, hand.current_gesture, hand.handmap)
            gu.draw_trajectory(frame, hand.positions_saved)
        
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
