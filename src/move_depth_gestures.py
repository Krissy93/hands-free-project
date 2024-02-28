#! /usr/bin/env python

from __future__ import division
import os
# Command needed to print only warnings and errors,
# otherwise the terminal would be filled with info logs from camera
os.environ['GLOG_minloglevel'] = '2'
import rospy
import numpy as np
import cv2
import utils


def gesture(points, chainH, chainU, chainD, chvalue, K, R, t, Debug=False):
    ''' Function to check which gesture is performed.
    Gestures defined:
    UP: corresponds to handmap = [0,1,1,0,0] with distant fingers
    DOWN: corresponds to handmap = [0,1,1,0,0] with close fingers
    NO GESTURE: none of the above '''

    # obtains the current handmap
    handmap = utils.closed_finger(points)

    # distance from fingers to be considered distant (in meters)
    dist = 0.08

    # if all values of handmap are equal to 1, then the gesture is HAND OPEN
    if handmap.count(1) == len(handmap):
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.GREEN + 'HAND OPEN' + utils.Color.END)
        G = 'HAND OPEN'

        chainH = chainH + 1
        chainD = 0
        chainU = 0

        if chainH == chvalue:
            if not Debug:
                rospy.loginfo(utils.Color.BOLD + utils.Color.GREEN + 'HAND OPEN' + utils.Color.END)

    elif handmap == [0, 1, 1, 0, 0]:
        # add z to index/middle points to calculate px2meters
        index_point = np.array([float(points[8][0]), float(points[8][1]), 1.0])
        middle_point = np.array([float(points[12][0]), float(points[12][1]), 1.0])
        index_point_m = utils.px2meters(index_point, K, R, t)
        middle_point_m = utils.px2meters(middle_point, K, R, t)
        dist_now = np.linalg.norm(index_point_m - middle_point_m)

        # if the index and middle finger value are equal to 1 and distance is small, then gesture is DOWN
        if dist_now < dist:
            if Debug:
                rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'DOWN' + utils.Color.END)
            G = 'DOWN'

            chainD = chainD + 1
            chainU = 0
            chainH = 0

            if chainD == chvalue:
                if not Debug:
                    rospy.loginfo(utils.Color.BOLD + utils.Color.BLUE + 'DOWN' + utils.Color.END)

    # if the index and middle finger value are equal to 1 and distance is big, then gesture is UP
        else:
            if Debug:
               rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'UP' + utils.Color.END)
            G = 'UP'

            chainU = chainU + 1
            chainD = 0
            chainH = 0

            if chainU == chvalue:
                if not Debug:
                    rospy.loginfo(utils.Color.BOLD + utils.Color.BLUE + 'UP' + utils.Color.END)

    else:
        # no gesture was detected

        # we accept the possible noises of the detection, thus we do not empty
        # the chain queue when no gesture is detected. A stronger constraint could be
        # to empty the chain in this case, and only accept values if the chain is not broken
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.PURPLE + 'NO GESTURE' + utils.Color.END)
        G = 'NO GESTURE'

    return chainH, chainU, chainD, G, str(handmap)


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
    rospy.init_node('hands_freeZ_node')

    ### VARIABLES

    # program flags and parameters that should not be changed
    nPoints = 22
    chainH = 0
    chainU = 0
    chainD = 0

    G = 'INIT'
    H = 'INIT'

    # depth increment
    incr = utils.incr

    ## PARAMETERS THAT MAY BE CHANGED BY THE USER (in code or from launch file)

    # draw value: set True to show skeleton, info, points, ecc. on frame, False otherwise
    draw = True

    # Caffe network threshold: if higher, less keypoints would be accepted
    threshold = 0.3

    # chain value: if lower, less finger positions are acquired before accepting gesture,
    # meaning that the reaction time is faster but may be less accurate due to disturbances
    chvalue = 2

    # path value: complete path of the src workspace folder from root
    # path = "/home/fole/sawyer_ws/src"
    path="/home/optolab/hands_free_workspace/src/hands_free2/src"

    # path_openpose value: complete path of the openpose/model folder from root
    path_openpose = "/home/optolab/openpose/models"

    # undistort value: set True to use undistorted image, False for the original one
    undistort = False

    # debug value: set True for debug logs, False otherwise
    Debug = False

    ###### STEP 1: INITIALIZATION

    # Caffe network initialization
    net = utils.init_network(path_openpose)

    # loading of the calibration parameters of both camera and robot
    # K, D, R, t, Rd, td = utils.loadcalibcamera(path+'/hands_free/src/yaml/camera_calibration.yaml')
    K, D, R, t = utils.loadcalibcamera("/home/optolab/hands_free_workspace/src/hands_free2/src/yaml")
    # RRobot = utils.loadcalibrobot(path+'/hands_free/src/yaml/robot_workspace_calibration.yaml')
    RRobot = utils.loadcalibrobot("/home/optolab/hands_free_workspace/src/hands_free2/src/yaml")
    # finds the coordinates of the reference point 0 in workspace H, used
    # to refer all the other points in the same workspace
    # if undistort:
    #     reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), Rd, td, K, D)
    # else:
    #     reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), R, t, K, D)
    #
    # reference = reference.flatten()
    # pt0 = (int(round(reference[0], 2)), int(round(reference[1], 2)))
    # reference = np.array([[reference[0], reference[1], 1.0]])
    # if undistort:
    #     Ref = utils.px2meters(reference, K, Rd, td)
    # else:
    #     Ref = utils.px2meters(reference, K, R, t)

    Ref = (0.0, 0.0)

    if Debug:
        # verify W dimensions/errors
        # if undistort:
        #     B3 = (42, 338)
        #     A1 = (481, 48)
        #     A3 = (485, 334)
        #
        #     pt1 = utils.px2meters(np.array([[float(B3[0]), float(B3[1]), 1.0]]), K, Rd, td)
        #     pt2 = utils.px2meters(np.array([[float(A3[0]), float(A3[1]), 1.0]]), K, Rd, td)
        #     pt3 = utils.px2meters(np.array([[float(A1[0]), float(A1[1]), 1.0]]), K, Rd, td)
        # else:
        #     # B3 = (44, 342)
        #     # A1 = (488, 46)
        #     # A3 = (496, 334)
        #     B3 = (46, 0)
        #     A1 = (0, 70)
        #     A3 = (46, 70)
        #
        #     pt1 = utils.px2meters(np.array([[float(B3[0]), float(B3[1]), 1.0]]), K, R, t)
        #     pt2 = utils.px2meters(np.array([[float(A3[0]), float(A3[1]), 1.0]]), K, R, t)
        #     pt3 = utils.px2meters(np.array([[float(A1[0]), float(A1[1]), 1.0]]), K, R, t)

        pt1 = (0.46, 0.0)
        pt2 = (0.0, 0.70)
        pt3 = (0.46, 0.70)
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
        # starts the camera object acquisition
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
            if undistort:
                chainH, chainU, chainD, G, H = gesture(points, chainH, chainU, chainD, chvalue, K, Rd, td, Debug)
            else:
                chainH, chainU, chainD, G, H = gesture(points, chainH, chainU, chainD, chvalue, K, R, t, Debug)

            ###### STEP 4: ROBOT MOVEMENT
            if chainH == chvalue:
                # move to home point (o0)
                robot.move_to_cartesian(utils.home[0] - 0.05, utils.home[1], utils.home[2] - utils.z_offset, time=0.00001, steps=1, Debug=Debug)
                if Debug:
                    print(utils.Color.BOLD + utils.Color.GREEN + ' -- ROBOT HOME POSE REACHED -- \n' + utils.Color.END)

            if chainU == chvalue:
                robot.increase_x(-incr, Debug=Debug)
                chainU = 0

            if chainD == chvalue:
                robot.increase_x(incr, Debug=Debug)
                chainD = 0

        ###### STEP 5: VISUALIZATION
        frameC = frame.copy()
        utils.draw_skeleton(frameC, points, draw)
        utils.draw_workspace(frameC, pt0, draw, undistort)
        utils.draw_gesture_info(frameC, inference_time, G, H, draw)
        cv2.imshow('Gesture detection', frameC)

        # catch q to close program
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break

    camera.stop()
    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()
