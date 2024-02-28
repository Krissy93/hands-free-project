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
import cartesian
import conversion_utils as conv
###
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
# sawyer interface
import intera_interface


def interpolate(px_points, debug=False):
    ''' Function that interpolates the list of acquired points in order to create
    a smoothed curve. It then samples the curve to obtain the points that will be sent
    to the robot controller.

    INPUTS:
    - px_points: list of pixel points corresponding to several index finger positions
                 creating the trajectory curve
    - debug: boolean flag to activate debugging functions

    OUTPUTS:
    - interpolated_points: list of lists containing pixel coordinates of trajectory points
                           in homogeneous coordinates [[x1,y1,1],[x2,y2,1],...]
    '''

    x = []
    y = []
    for p in px_points:
        x.append(p[0])
        y.append(p[1])
    ###### todo: check if px_points is a list of lists [[x,y],[x,y]] or list of tuples
    ###### to change these lines accordingly. Basically I want an array of N rows and 2 cols

    points = np.array([x, y]).T

    # creates the cumulative sum of points differences, basically doing:
    # np.diff along y -> points[0][i] - points[1][i]
    # power of 2 of the resulting matrix
    # sum along x of the resulting matrix, thus summing rows elements together
    # squared root of resulting vector
    # cumulative sum of resulting vector, thus obtaining (a[0], a[0]+a[1], a[0]+a[1]+a[2]...)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    # adds a 0 at position 0 of the resulting vector, then normalize the result
    # for the last value which is the cumulative sum of elements
    distance = np.insert(distance, 0, 0) / distance[-1]

    # sets the parameter k of spline polynomial weight. If an adequate number of points
    # is present, choose k=5, otherwise choose it equal to number of points - 1
    if len(points) < 6:
        k = len(points) - 1
    else:
        k = 5

    # Build a list of the spline function, one for each dimension
    # the univariate spline computes a spline given a set x of incremental points, basically
    # we need to pass it a set of points in range 0-1. The y is an array of equal length of
    # values corresponding to the x ones. Parameter k is the polynomial weight of spline,
    # default is cubic with k = 3. We choose k=len(points)-1 when the number of points in trajectory
    # is less than 6 to avoid interpolation errors, otherwise we set it to the max value.
    # s is the smoothing parameter that makes the spline more or less sharp by choosing
    # a certain number of knots until the s condition is satisfied. More details here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    splines = [UnivariateSpline(distance, coords, k=k, s=0.2) for coords in points.T]

    # creates equally spaced points using linspace that will be used as x for the spline function
    # to compute corresponding y. Number of points in range 0-1 to create is len of points x2
    alpha = np.linspace(0, 1, len(points)*2)
    # for each spline computes the y by using alpha points, then stacks them vertically
    # and finally transpose them. At the end we'll have N rows and 2 cols of x,y points
    points_fitted = np.vstack([spl(alpha) for spl in splines]).T

    # plot both original and interpolated curves to show differences of
    # spline interpolation vs original points
    if debug:
        a1, = plt.plot(*points.T)
        a2, = plt.plot(*points_fitted.T)
        plt.axis('equal')
        plt.legend([a1, a2],['original', 'fitted'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # converts to list the transpose of the resulting points_fitted array
    points_fitted_list = points_fitted.T.tolist()
    # creates a list of interpolated points initialized with one element, which is x, y of
    # the first point in homogeneous coordinates
    interpolated_points = [[points_fitted_list[0][0], points_fitted_list[1][0], 1.0]]
    # iteratively fills the list with other points
    for i in range(1, len(points_fitted)-1):
        # first checks distance between last point in list and current point in points_fitted
        # ((xN - x)^2) + (yN - y)^2)^0.5
        dist = math.sqrt((interpolated_points[-1][0] - points_fitted_list[0][i]) ** 2 + (interpolated_points[-1][1] - points_fitted_list[1][i]) ** 2)

        # DO NOT save points that are too close according to threshold (values are in meters)
        if dist > 0.0175:
            interpolated_points.append([points_fitted_list[0][i], points_fitted_list[1][i], 1.0])

    # finally, adds the final point in original pixel points list px_points
    interpolated_points.append([px_points[-1][0], px_points[-1][1], 1.0])

    return interpolated_points

def hand_open_action(hand, robot_home, robot_orientation, linear_speed):
    rospy.loginfo(gu.Color.BOLD + gu.Color.GREEN + 'HAND OPEN: RESETTING TRAJECTORY' + gu.Color.END)
    # HAND OPEN ACTION: resets points lists and sets acquire to true
    hand.acquire = True
    hand.positions_saved = []
    hand.index_positions = []

    # also resets robot position to home
    cartesian.move2cartesian(position=robot_home, orientation=robot_orientation, in_tip_frame=True, linear_speed=linear_speed)
    print(gu.Color.BOLD + gu.Color.GREEN + ' -- ROBOT HOME POSE REACHED -- ' + gu.Color.END)

def move_action(hand, depth, robot_points, orientation, linear_speed):
    rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'READY TO MOVE' + gu.Color.END)
    # resets acquire flag and move chain
    hand.acquire = False
    hand.chain_move = 0

    # compute interpolated points if more than one is present
    if len(robot_points) > 1:
        interpolated_points = interpolate(robot_points)
    else:
        interpolated_points = robot_points

    rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + '-- MOVING... --' + gu.Color.END)

    for p in interpolated_points:
        # depth[0] may be 0, 1 or 2 corresponding to x, y or z coordinates
        p[depth[0]] = depth[1]
        # move to each point in a loop until end of trajectory
        cartesian.move2cartesian(position=tuple(p), orientation=orientation, in_tip_frame=True, linear_speed=linear_speed)

    rospy.loginfo(gu.Color.BOLD + gu.Color.GREEN + '-- MOVEMENT COMPLETED --' + gu.Color.END)

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

    # project point (0,0) to camera coordinates
    reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), R, t, K, D)
    reference = reference.flatten()
    # gets homogeneous coordinates of point
    reference = np.array([[reference[0], reference[1], 1.0]])
    # convert reference point to meters
    ref_pt = conv.px2meters(reference, K, R, t)

    return ref_pt

def main():
    ''' Main program of Hands-Free project.

    This program does the following:
    STEP 1: INITIALIZATION
            Some initialization procedures are carried out when the node is started,
            such as the OpenPose network initialization, the loading of the calibration
            parameters saved in the corresponding YAML files, the initialization of the
            camera object and robot etc.
    STEP 2: FRAME ACQUISITION
            Starts the camera and acquires the undistorted frame (applies coorections)
            that is passed to the Caffe network to detect the keypoints.
    STEP 3: GESTURE RECOGNITION
            According to the defined gestures and the detected keypoints, determines
            which gesture is present in the frame.
    STEP 4: ROBOT MOVEMENT
            Performs calculations to find the corresponding point from image frame
            to workspace H and from workspace H to robot coordinates. Interpolates
            the list of points and passes to the cartesian movement function in
            order to moves the robot.
    STEP 5: VISUALIZATION
            If the draw flag has been selected, shows the acquired frame and
            draws on that the skeleton, the workspace, the acquired points and
             some debug info useful for the user. '''

    # node creation
    rospy.init_node('hands_free_node')

    ## PARAMETERS THAT MAY BE CHANGED BY THE USER (in code or from launch file)

    # draw value: set True to show skeleton, info, points, ecc. on frame, False otherwise
    if rospy.has_param('~draw'):
        draw = rospy.get_param('~draw')
    else:
        draw = True

    # Caffe network threshold: if higher, less keypoints would be accepted
    if rospy.has_param('~threshold'):
         threshold = rospy.get_param('~threshold')
     else:
         threshold = 0.2

    # chain value: if lower, less finger positions are acquired before accepting gesture,
    # meaning that the reaction time is faster but may be less accurate due to disturbances
    if rospy.has_param('~max_chain'):
         max_chain = rospy.get_param('~max_chain')
     else:
         max_chain = 3

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
         depth_val = 0.7
    depth = [depth_coord, depth_val]

    # path_openpose value: complete path of the openpose/model folder from root
    if rospy.has_param('~path_openpose'):
         path_openpose = rospy.get_param('~path_openpose')
     else:
         path_opH2Renpose = '/home/optolab/openpose/models'

    # path_openpose value: complete path of the openpose/model folder from root
    if rospy.has_param('~undistort'):
         undistort = rospy.get_param('~undistort')
     else:
         undistort = False

    # robot parameters such as home coordinates, tip orientation and speed
    if rospy.has_param('~robot_home'):
         robot_home = tuple(rospy.get_param('~robot_home'))
     else:
         robot_home = (depth_val, 0.04, 0.27)

    if rospy.has_param('~robot_orientation'):
         robot_orientation = tuple(rospy.get_param('~robot_orientation'))
     else:
         robot_orientation = (0.5, 0.5, 0.5, 0.5)

    if rospy.has_param('~robot_speed'):
         robot_speed = rospy.get_param('~robot_speed')
     else:
         robot_speed = 0.3

    if rospy.has_param('~camera'):
         cam_name = rospy.get_param('~camera')
     else:
         cam_name = 'Kinect'

    # debug value: set True for debug logs, False otherwise
    if rospy.has_param('~debug'):
         debug = rospy.get_param('~debug')
     else:
         debug = False

    path = os.path.realpath(os.getcwd())

    ###### STEP 1: INITIALIZATION

    # Hand object initialization
    hand = hgu.Hand()

    # loading of the calibration parameters of both camera and robot
    camera_calibration = utils.yaml2dict(path+'/yaml/camera_calibration.yaml')
    K = camera_calibration['K']
    D = camera_calibration['D']
    R = camera_calibration['R']
    t = camera_calibration['t']

    workspace_calibrations = utils.yaml2dict(path+'/yaml/robot_workspace_calibration.yaml')
    R_H2W = workspace_calibrations['H2WCalibration']
    R_W2R = workspace_calibrations['W2RCalibration']
    # moves robot to home position
    rospy.loginfo(gu.Color.BOLD + gu.Color.GREEN + '-- INITIALIZING ROBOT, MOVING TO HOME --' + gu.Color.END)
    cartesian.move2cartesian(position=robot_home, orientation=robot_orientation, in_tip_frame=True, linear_speed=robot_speed)

    # computes reference point
    ref_pt = get_ref_point(K, D, R, t)

    if cam_name == 'Kinect':
        camera = utils.Kinect(enable_rgb=True, enable_depth=False, need_bigdepth=False, need_color_depth_map=False, K=K, D=D)
    else:
        camera = utils.Camera(enable_rgb=True)

    # Robot object creation
    #robot = utils.Robot()

    while not rospy.is_shutdown():
        ###### STEP 2: FRAME ACQUISITION
        # starts the camera object acquisition and eventually clear buffer
        camera.acquire(False)
        # calls the undistortion method: the frames are UNDISTORTED from now on,
        # so the calibration matrixes that must be used are Rd and td!!
        frame = camera.RGBundistorted.copy()

        #frame, b3, a1, a3 = utils.correzione_prospettica(B3, A1, A3, reference, R, t, K, D, frame.copy())

        # detects the hand keypoints
        hand.inference(frame)
        if debug:
            rospy.loginfo(gu.Color.BOLD + gu.Color.GREEN + 'points: ' + str(hand.points) + gu.Color.END)

        ###### STEP 3: GESTURE RECOGNITION
        # if all points are None or the reference keypoint is None, there is no gesture
        # so it skips the detection functions and assigns a no gesture handle to current_gesture
        if all(x == None for x in self.points[1:]) or self.points[0] == None:
            hand.current_gesture = 'NO GESTURE'
            if debug:
                rospy.loginfo(gu.Color.BOLD + gu.Color.RED + 'NO GESTURE FOUND IN IMAGE' + gu.Color.END)

        # else, it finds the correct gesture based on the keypoints detected
        else:
            hand.get_gesture()

            ###### STEP 4: ROBOT MOVEMENT
            # according to gesture type identified, performs a robot action
            # please note that the update of positions_saved happens inside get_gesture()
            if hand.current_gesture == 'HAND OPEN':
                hand_open_action(hand, robot_home, robot_orientation, linear_speed)
                hand.current_gesture = 'NO GESTURE'

            elif hand.current_gesture = 'MOVE':
                # points_list, K, R, t, R_H2W, R_W2R, depth, ref_pt, debug=False
                robot_points = cu.px2R(hand.positions_saved, K, R, t, R_H2W, R_W2R, depth, ref_pt, debug)
                move_action(hand, depth, robot_points, robot_orientation, linear_speed)
                hand.current_gesture = 'NO GESTURE'

        ###### STEP 5: VISUALIZATION
        gu.draw_openpose_skeleton(frame, hand.points)
        #gu.draw_workspace(frame, markers=workspace_calibrations['Master_H'])
        gu.draw_gesture_info(frame, hand.inference_time, hand.current_gesture, hand.handmap)
        gu.draw_trajectory(frame, to_move)
        cv2.imshow('Gesture and trajectory detection', frame)

        # when 'q' is pressed, closes program
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break

    # stops camera
    camera.stop()
    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()