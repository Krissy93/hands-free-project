
#! /usr/bin/env python

from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import datetime
import math
# Command needed to print only warnings and errors,
# otherwise the terminal would be filled with info logs from camera
os.environ['GLOG_minloglevel'] = '2'
import rospy
import numpy as np
import cv2
###
import utils
import graphical_utils as gu
import hand_gesture_utils as hgu
#import cartesian
import conversion_utils as cu
###
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from tf.transformations import quaternion_from_euler


def get_ref_point(K, D, R, t):
    ''' Finds the coordinates of the reference point 0 in workspace H, used
    to refer all the other points in the same workspace.

    INPUTS:
    - K: camera matrix
    - D: camera distortion parameters
    - R: camera calibration matrix (rotation)
    - t: camera calibration vector (translation)

    OUTPUTS:
    - ref_pt: reference point corresponding to (0,0) of workspace H in meters
    '''

    # Converti K, D, R e t in array NumPy se non lo sono già
    K = np.array(K, dtype=np.float64)
    D = np.array(D, dtype=np.float64)
    R = np.array(R, dtype=np.float64)
    t = np.array(t, dtype=np.float64)

    # project point (0,0) to camera coordinates
    reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), R, t, K, D)
    reference = reference.flatten()
    # gets homogeneous coordinates of point
    reference = np.array([[reference[0], reference[1], 1.0]])
    rospy.loginfo(reference)
    # convert reference point to meters
    ref_pt = cu.px2meters(reference, K, R, t)

    return ref_pt, reference

def main():
    # node creation
    rospy.init_node('hands_free_node')


    # x value: this is the value that we set as fixed because we control the robot positioning
    # only on plane ZY, thus we fix the depth (in our case X) according to workspace W
    # distance from the robot (to avoid accidental collisions)
    if rospy.has_param('~depth_coord'):
         depth_coord = rospy.get_param('~depth_coord')
    else:
         depth_coord = 'x'
    if rospy.has_param('~depth_val'):
         depth_val = rospy.get_param('~depth_val')
    else:
         depth_val = -0.25
    # Mappa le coordinate x, y, z a indici numerici
    coord_mapping = {'x': 0, 'y': 1, 'z': 2}

    # Assicurati che depth_coord sia uno degli assi e mappalo a un intero
    if depth_coord in coord_mapping:
        depth = [coord_mapping[depth_coord], depth_val]
    else:
        rospy.logwarn("Invalid depth_coord value: {}".format(depth_coord))
        depth = [0, depth_val]  # Default to 'x' if the value is invalid


    ###### STEP 1: INITIALIZATION
    # loading of the calibration parameters of both camera and robot
    camera_calibration = utils.yaml2dict('/home/jacopo/URProject/src/hands-free-project/src/yaml/camera_calibration.yaml')
    K = camera_calibration['K']
    D = camera_calibration['D']
    R = camera_calibration['R']
    t = camera_calibration['t']

    workspace_calibrations = utils.yaml2dict('/home/jacopo/URProject/src/hands-free-project/src/yaml/calibration.yaml')
    R_H2W = workspace_calibrations['H2WCalibration']
    #R_H2W = workspace_calibrations['H2W_2']

    # computes reference point
    debug = True
    ref_pt, ref_px = get_ref_point(K, D, R, t)

    print(ref_pt)

    saved_points = [[1424,271,1],[1135,263,1],[854,253,1],[1408,472,1],[1124,461,1],[848,449,1],[1395,668,1],[1114,652,1],[842,640,1],[1448,257,1]]
        
    rospy.loginfo(gu.Color.BOLD + gu.Color.RED + f'Saved positions: {saved_points}'+ gu.Color.END)

    robot_points = cu.px2R(saved_points, K, R, t, R_H2W, depth, ref_pt, debug)

    rospy.loginfo(gu.Color.BOLD + gu.Color.RED + f'Robot_points: {robot_points}'+ gu.Color.END)


if __name__ == '__main__':
    main()