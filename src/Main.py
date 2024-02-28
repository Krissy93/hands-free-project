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
import utils
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def H2R(Ph, Robot, x, Debug=False):
    ''' Function to properly convert a given point (in meters) from workspace
    H to workspace W. The obtained robot position is used to move the robot in that point.
    Please note that moving a robot in cartesian coordinates could lead to interpolation
    errors depending on the point and on the robot itself. It is also a good practice to
    move the robot in its neutral position/home position at the startup of the program. '''

    # flatten the given point and transform it in homogeneous coordinates
    # since we use place ZY instead of XY we must give to the function the Y first and the X second!
    Ph = Ph.flatten()
    Ph = np.array([[Ph[1], Ph[0], 1.0]])
    if Debug:
        rospy.loginfo(utils.Color.BOLD + utils.Color.PURPLE + 'Point is: ' + str(Ph) + utils.Color.END)

    # transform the point using the robot rototranslation matrix: (3x3) * (3x1)
    Pr = Robot.dot(Ph.T)
    if Debug:
        rospy.loginfo(utils.Color.BOLD + utils.Color.PURPLE + 'Calculated point: ' + str(Pr) + utils.Color.END)

    return Pr


def px2R(pixel_points, K, R, t, Robot, x, Ref):
    ''' Function to convert a list of points (in pixel) in the corresponding robot workspace's points (in meters). '''

    robot_points = []
    for p in pixel_points:
        Ph = utils.px2meters(p, K, R, t)
        # finds the coordinates of the calculated point with respect to reference point
        new = (Ph - Ref)

        # calculates robot coordinates from starting point new in reference system H
        # if workspace H and W differ, you need to calibrate them too (in our case this was not
        # necessary because the two had the same size but different orientation)
        robot_points.append(H2R(new, Robot, x))

    return robot_points


def draw_trajectory(frame, to_move, draw):
    ''' Function to draw the points acquired for calculate the trajectory.
    Does not return anything. '''

    # draw trajectory on frame if the draw flag has been set as True
    if draw:
        for point in to_move:
            P = (int(point[0][0]), int(point[0][1]))
            cv2.circle(frame, P, 2, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)


def gesture(points, chainH, chainI, chainM, acquire, to_move, chvalue, Debug=False):
    ''' Function to check which gesture is performed.
    Gestures defined:
    HAND OPEN: corresponds to handmap = [1,1,1,1,1], all fingers opened
    INDEX: corresponds to handmap = [0,1,0,0,0], only index finger opened
    MOVE: corresponds to handmap = [0,1,1,0,0], index and middle fingers opened
    NO GESTURE: none of the above '''

    # obtains the current handmap
    handmap = utils.closed_finger(points)

    # if all values of handmap are equal to 1, then the gesture is HAND OPEN
    if handmap.count(1) == len(handmap):
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.GREEN + 'HAND OPEN' + utils.Color.END)
        G = 'HAND OPEN'

        chainH = chainH + 1
        chainM = 0

        if chainH == chvalue:
            to_move = []
            acquire = True
            chainI = []
            if not Debug:
                rospy.loginfo(utils.Color.BOLD + utils.Color.GREEN + 'HAND OPEN' + utils.Color.END)

    # if only the index finger value is equal to 1, then the gesture is INDEX
    elif handmap == [0, 1, 0, 0, 0]:
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'INDEX' + utils.Color.END)
        G = 'INDEX'

        chainH = 0
        chainM = 0

        # if the acquire flag has been set to True, the length of the chain
        # is less than the defined chain value and the index finger keypoint exists,
        # then it appends the index finger keypoint coordinates to the chain
        if acquire and len(chainI) < chvalue and points[8]:
            chainI.append(points[8])

            if len(chainI) == chvalue:
                if not Debug:
                    rospy.loginfo(utils.Color.BOLD + utils.Color.BLUE + 'INDEX' + utils.Color.END)
                # zips x coordinates in one sublist and y coordinats in another sublist
                rechain = zip(*chainI)
                # calculate the mean
                mean = np.array([[sum(rechain[0]) / len(rechain[0]), sum(rechain[1]) / len(rechain[1]), 1.0]])

                if len(to_move) != 0:
                    last = to_move[-1]
                    dist = math.sqrt((last[0][0] - mean[0][0])**2 + (last[0][1] - mean[0][1])**2)
                    # don't save points too close together (value in px)
                    if dist > 5:
                        to_move.append(mean)
                else:
                    to_move.append(mean)

                if Debug:
                    rospy.loginfo(utils.Color.BOLD + utils.Color.YELLOW + 'CHAIN VALUE REACHED. MEAN IS: ' + str(mean[0][0]) + ', ' + str(mean[0][1]) + utils.Color.END)
                # empty the queue
                chainI = []

    # if the index and middle finger value are equal to 1, then the gesture is MOVE
    elif handmap == [0, 1, 1, 0, 0]:
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'MOVE' + utils.Color.END)
        G = 'MOVE'

        chainM = chainM + 1
        chainH = 0

        if acquire and chainM == chvalue:
            if not Debug:
                rospy.loginfo(utils.Color.BOLD + utils.Color.BLUE + 'MOVE' + utils.Color.END)
            acquire = False

    else:
        # no gesture was detected

        # we accept the possible noises of the detection, thus we do not empty
        # the chain queue when no gesture is detected. A stronger constraint could be
        # to empty the chain in this case, and only accept values if the chain is not broken
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.PURPLE + 'NO GESTURE' + utils.Color.END)
        G = 'NO GESTURE'

    return chainH, chainI, chainM, acquire, to_move, G, str(handmap)


def interpolate(index_points, Debug=False):
    ''' Function that interpolates the list of acquired points in order to create a more gentle curve.
    It then samples the curve to obtain the points for the move_to_cartesian function'''

    x = []
    y = []
    for p in index_points:
        x.append(p[0])
        y.append(p[1])

    points = np.array([x, y]).T[0]
    # Linear length along the line
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))

    distance = np.insert(distance, 0, 0) / distance[-1]

    if len(points) < 6:
        k = len(points) - 1
    else:
        k = 5

    # Build a list of the spline function, one for each dimension
    splines = [UnivariateSpline(distance, coords, k=k, s=0.2) for coords in points.T]

    # Computed the spline for the asked distances (sample)
    alpha = np.linspace(0, 1, len(points)*2)

    points_fitted = np.vstack(spl(alpha) for spl in splines).T

    # plot both original and interpolated curves
    if Debug:
        plt.plot(*points.T)
        plt.plot(*points_fitted.T)
        plt.axis('equal')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    interpolated_points = []
    points_fitted_list = points_fitted.T.tolist()
    interpolated_points.append([points_fitted_list[0][0], points_fitted_list[1][0], 1.0])
    for i in range(1, len(points_fitted)-1):
        dist = math.sqrt((interpolated_points[-1][0] - points_fitted_list[0][i]) ** 2 + (interpolated_points[-1][1] - points_fitted_list[1][i]) ** 2)

        # don't save points too close together (value in meters)
        if dist > 0.0175:
            interpolated_points.append([points_fitted_list[0][i], points_fitted_list[1][i], 1.0])
    interpolated_points.append([index_points[-1][0], index_points[-1][1], 1.0])

    return interpolated_points


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

    ### VARIABLES

    # program flags and parameters that should not be changed
    nPoints = 22
    acquire = False
    chainH = 0
    chainI = []
    chainM = 0
    to_move = []

    G = 'INIT'
    H = 'INIT'

    ## PARAMETERS THAT MAY BE CHANGED BY THE USER (in code or from launch file)

    # draw value: set True to show skeleton, info, points, ecc. on frame, False otherwise
    draw = True
    if rospy.has_param('~draw'):
        draw = rospy.get_param('~draw')

    # Caffe network threshold: if higher, less keypoints would be accepted
    threshold = 0.3
    if rospy.has_param('~threshold'):
        threshold = rospy.get_param('~threshold')

    # chain value: if lower, less finger positions are acquired before accepting gesture,
    # meaning that the reaction time is faster but may be less accurate due to disturbances
    chvalue = 3
    if rospy.has_param('~chvalue'):
        chvalue = rospy.get_param('~chvalue')

    # x value: this is the value that we set as fixed because we control the robot positioning
    # only on plane ZY, thus we fix the depth (in our case X) according to workspace W
    # distance from the robot (to avoid accidental collisions)
    x = 0.7
    if rospy.has_param('~x'):
        x = rospy.get_param('~x')

    # path value: complete path of the hands_free folder from root
    path = "/home/fole/sawyer_ws/src"
    if rospy.has_param('~path'):
        path = rospy.get_param('~path')

    # path_openpose value: complete path of the openpose/model folder from root
    path_openpose = "/home/fole/openpose/models"
    if rospy.has_param('~path_openpose'):
        path_openpose = rospy.get_param('~path_openpose')

    # undistort value: set True tu use undistorted image, False for the original one
    undistort = True
    if rospy.has_param('~undistort'):
        undistort = rospy.get_param('~undistort')

    # debug value: set True for debug logs, False otherwise
    Debug = False
    if rospy.has_param('~debug'):
        Debug = rospy.get_param('~debug')

    ###### STEP 1: INITIALIZATION
    # prepare debug file for storing points
    now = datetime.datetime.now()
    f = open(path + "/hands_free/src/debug/points.txt", "a")
    f.write("----- ")
    f.write(now.strftime("%d-%m-%Y %H:%M:%S"))
    f.write(" -----\n\n")
    f.close()

    # Caffe network initialization
    net = utils.init_network(path_openpose)

    # loading of the calibration parameters of both camera and robot
    K, D, R, t, Rd, td = utils.loadcalibcamera(path+'/hands_free/src/yaml/camera_calibration.yaml')
    RRobot = utils.loadcalibrobot(path+'/hands_free/src/yaml/robot_workspace_calibration.yaml')

    # finds the coordinates of the reference point 0 in workspace H, used
    # to refer all the other points in the same workspace
    if undistort:
        reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), Rd, td, K, D)
    else:
        reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), R, t, K, D)

    reference = reference.flatten()
    pt0 = (int(round(reference[0], 2)), int(round(reference[1], 2)))
    reference = np.array([[reference[0], reference[1], 1.0]])
    if undistort:
        Ref = utils.px2meters(reference, K, Rd, td)
    else:
        Ref = utils.px2meters(reference, K, R, t)

    if Debug:
        # verify W dimensions/errors
        if undistort:
            B3 = (42, 338)
            A1 = (481, 48)
            A3 = (485, 334)

            pt1 = utils.px2meters(np.array([[float(B3[0]), float(B3[1]), 1.0]]), K, Rd, td)
            pt2 = utils.px2meters(np.array([[float(A3[0]), float(A3[1]), 1.0]]), K, Rd, td)
            pt3 = utils.px2meters(np.array([[float(A1[0]), float(A1[1]), 1.0]]), K, Rd, td)
        else:
            B3 = (44, 342)
            A1 = (488, 46)
            A3 = (496, 334)

            pt1 = utils.px2meters(np.array([[float(B3[0]), float(B3[1]), 1.0]]), K, R, t)
            pt2 = utils.px2meters(np.array([[float(A3[0]), float(A3[1]), 1.0]]), K, R, t)
            pt3 = utils.px2meters(np.array([[float(A1[0]), float(A1[1]), 1.0]]), K, R, t)

        new1 = (pt1 - Ref)
        new2 = (pt2 - pt1)
        new3 = (pt3 - pt2)
        new4 = (Ref - pt3)
        print('Distance B1-B3: ' + str(new1))
        print('Distance B3-A3: ' + str(new2))
        print('Distance A3-A1: ' + str(new3))
        print('Distance A3-B1: ' + str(new4))

    # Camera object creation
    camera = utils.Camera(True, K, D)

    # Robot object creation
    robot = utils.Robot()

    while not rospy.is_shutdown():
        ###### STEP 2: FRAME ACQUISITION
        # starts the camera object acquisition and eventually clear buffer
        if not acquire:
            camera.grab()

        camera.acquire(True)

        # choose original or undistorted frame
        if undistort:
            frame = camera.RGBundistorted.copy()
        else:
            frame = camera.color_new.copy()

        # crop the frame
        frame = utils.crop(frame)

        # detects the hand keypoints
        points, inference_time = utils.hand_keypoints(net, frame, threshold, nPoints)
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.GREEN + 'points, inference_time: ' + str(points) + "    " + str(inference_time) + utils.Color.END)

        ###### STEP 3: GESTURE RECOGNITION
        # if all points are None or the reference keypoint is None, there is no gestures
        if all(x == None for x in points[1:]) or points[0] == None:
            if Debug:
                rospy.loginfo(utils.Color.BOLD + utils.Color.RED + 'NO GESTURE FOUND IN IMAGE' + utils.Color.END)

            G = 'NO GESTURE'
            H = 'NONE'

        # else, it finds the correct gesture based on the keypoints detected
        else:
            chainH, chainI, chainM, acquire, to_move, G, H = gesture(points, chainH, chainI, chainM, acquire, to_move, chvalue)

            ###### STEP 4: ROBOT MOVEMENT
            if chainH == chvalue:
                # move to home point (origin)
                home = [x, 0.0, 1.105]
                robot.move_to_cartesian(home[0] - 0.05, home[1], home[2] - 0.93, time=0.00001, steps=1)
                print(utils.Color.BOLD + utils.Color.GREEN + ' -- ROBOT HOME POSE REACHED -- ' + utils.Color.END)

            if not acquire and chainM == chvalue:

                if undistort:
                    robot_points = px2R(to_move, K, Rd, td, RRobot, x, Ref)
                else:
                    robot_points = px2R(to_move, K, R, t, RRobot, x, Ref)

                if len(robot_points) > 1:
                    interpolated_points = interpolate(robot_points)
                else:
                    interpolated_points = robot_points
                    
                chainM = 0

                rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + '-- MOVING... --' + utils.Color.END)

                for Pr in interpolated_points:
                    robot.move_to_cartesian(x, round(Pr[0], 2), round(Pr[1], 2), time=0.00001, steps=1)

                    # save points for debug
                    f = open("/home/fole/sawyer_ws/src/hands_free/src/debug/points.txt", "a")
                    f.write(str(Pr))
                    f.write("\n")
                    f.close()

                print(utils.Color.BOLD + utils.Color.GREEN + '-- MOVEMENT COMPLETED --' + utils.Color.END)

                # save points for debug
                f = open("/home/fole/sawyer_ws/src/hands_free/src/debug/points.txt", "a")
                f.write("\n\n\n\n")
                f.close()

        ###### STEP 5: VISUALIZATION
        frameC = frame.copy()
        utils.draw_skeleton(frameC, points, draw)
        utils.draw_workspace(frameC, pt0, draw, undistort)
        utils.draw_gesture_info(frameC, inference_time, G, H, draw)
        draw_trajectory(frameC, to_move, draw)
        cv2.imshow('Gesture and trajectory detection', frameC)

        # catch q to close program
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break

    camera.stop()
    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()
