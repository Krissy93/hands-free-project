#!/usr/bin/env python

from __future__ import division
import os
# Command needed to print only warnings and errors,
# otherwise the terminal would be filled with info logs from camera
os.environ['GLOG_minloglevel'] = '2'
import rospy
import time
import math
import yaml # in python 2.7 be sure to use pip install pyyaml==5.1

import numpy as np
from scipy.spatial import distance
import cv2
import sys
import caffe
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from geometry_msgs.msg import Pose
#from cartesian import *
# sawyer interface
#import intera_interface
# custom made utility file
from utils import *

################


def init_network():
    ''' Function to initialize network parameters.
    Be sure to place the network and the weights in the correct folder. '''

    rospy.loginfo(color.BOLD + color.YELLOW + 'INITIALIZING CAFFE NETWORK...' + color.END)
    protoFile = "HandPose/hand/pose_deploy.prototxt"
    weightsFile = "HandPose/hand/pose_iter_102000.caffemodel"

    net = caffe.Net(protoFile, 1, weights = weightsFile)
    caffe.set_mode_gpu()

    return net


def px2meters(pt, K, R, t):
    ''' Conversion function from pixels to meters used to obtain the match between
    the index position in the image frame (pixels) and the corresponding position in
    workspace H (meters). The returned point XYZ is in meters'''

    # find the inverse matrix K^-1
    K2 = np.linalg.inv(K)
    # find the inverse matrix R^-1
    R2 = np.linalg.inv(R)
    # transpose initial point. Be sure to pass it as [Xpx, Ypx, 1.0]
    pt = pt.T
    # STEP 1: K^-1 * point -> (3x3) * (3x1) = 3x1
    S = K2.dot(pt)
    # STEP 2: (K^-1 * point) - t -> (3x1) - (3x1) = 3x1
    N = S - t
    # STEP 3: R^-1 * ((K^-1 * point) - t) -> (3x3) * (3x1) = (3x1)
    XYZ = R2.dot(N)

    return XYZ


def H2R(Ph, Robot, x):
    ''' Function to properly convert a given point (in meters) from workspace
    H to workspace W. The obtained robot position is used to move the robot in that point.
    Please note that moving a robot in cartesian coordinates could lead to interpolation
    errors depending on the point and on the robot itself. It is also a good practice to
    move the robot in its neutral position/home position at the startup of the program. '''

    # flatten the given point and transform it in homogeneous coordinates
    # since we use place ZY instead of XY we must give to the function the Y first and the X second!
    Ph = Ph.flatten()
    Ph = np.array([[Ph[1], Ph[0], 1.0]])
    rospy.loginfo(color.BOLD + color.PURPLE + 'Point is: ' + str(Ph) + color.END)

    # transform the point using the robot rototranslation matrix: (3x3) * (3x1)
    Pr = Robot.dot(Ph.T)
    print(color.BOLD + color.PURPLE + 'Calculated point: ' + str(Pr) + color.END)

    #move2cartesian(position=(x, round(Pr[0],2), round(Pr[1],2)), orientation=(0.5, 0.5, 0.5, 0.5), in_tip_frame=True, linear_speed=0.3)

    return Pr


def hand_keypoints(net, frame, threshold, nPoints):
    ''' This function is used to perform INFERENCE in real time on each acquired frame.
    The Caffe network initialized at the startup of the program is passed to this function
    to perform the hand keypoint estimation. Using the given threshold, the predicted
    keypoints may be accepted or discarded. '''

    # we calculate the total inference time as a difference between the starting time
    # of this function and the time corresponding to the output creation
    before = time.time()
    # these values are needed by the model to correctly detect the keypoints!
    aspect_ratio = frame.shape[1]/frame.shape[0]
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # create the image blob for the model
    net.blobs['image'].reshape(1, 3, inHeight, inWidth)
    net.blobs['image'].data[...] = inpBlob
    # call the prediction method to find out the estimated keypoints
    output = net.forward()
    output = output['net_output']

    # gets inference time required by network to perform the detection
    inference_time = time.time() - before

    # Empty list to store the detected keypoints
    points = []
    for i in range(nPoints):
        # confidence map of corresponding keypoint location
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

        # find global maxima of the probMap
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # if the probability of the given keypoint is greater than the threshold,
        # it stores the point, else it appends a 'None' placeholder
        if prob > threshold:
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    return points, inference_time


def pointsW(lista, pt, Ref, K, Rd, td, RRobot):

    pt = np.array([[pt[0], pt[1], 1.0]])
    PT = px2meters(pt, K, Rd, td)
    new = (PT - Ref)
    PR = H2R(new, RRobot, 0.7)
    lista.append([(pt[0][0], pt[0][1]), (new[0].tolist(), new[1].tolist(), new[2].tolist()), (round(PR[0],2), round(PR[1],2))])

def draw_skeleton(frame, points, draw, inference_time, G, H, pt0, pt1, pt2, pt3):
    ''' Function to draw the skeleton of one hand according to
    a pre-defined pose pair scheme to the frame. Does not return anything. '''

    POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],
                  [10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    B2 = (315, 390)
    B3 = (318, 633)
    B4 = (506, 267)
    B5 = (506, 513)
    O1 = (698, 141)
    O0 = (696, 391)
    O2 = (693, 636)
    A4 = (890, 268)
    A5 = (886, 516)
    A1 = (1086, 140)
    A2 = (1081, 393)
    A3 = (1075, 643)

    # always use the copy() method when working on cv2 frames that need to be modified
    A = frame.copy()

    #A = find_marker_centroids(A)

    # draw skeleton on frame if the draw flag has been set as True
    if draw:
        for pair in POSE_PAIRS:
            # pose pairs represent the lines connecting two keypoints, used to correclty draw the skeleton
            partA = pair[0]
            partB = pair[1]

            # if there is a point in both keypoints of the pair, draws the point and the connected line
            if points[partA] and points[partB]:
                cv2.line(A, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(A, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(A, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # DRAW THE WORKSPACE
    # cv2.circle(A, pt0, 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    # cv2.circle(A, pt1, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.circle(A, pt2, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.circle(A, pt3, 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.line(A, pt0, pt1, (0, 0, 0), 2)
    # cv2.line(A, pt1, pt2, (0, 0, 0), 2)
    # cv2.line(A, pt2, pt3, (0, 0, 0), 2)
    # cv2.line(A, pt3, pt0, (0, 0, 0), 2)

    cv2.circle(A, pt0, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, A1, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, A2, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, A3, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, A4, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, A5, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, O1, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, O0, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, O2, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, B2, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, B3, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, B4, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.circle(A, B5, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    # the structure is: image, string of text, position from the top-left angle, font, size, BGR value of txt color, thickness, graphic
    cv2.putText(A, 'INFERENCE TIME: ' + str(inference_time) + ' SEC', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 65, 242), 3, cv2.LINE_AA)
    cv2.putText(A, str(G) + ' || HANDMAP: ' + str(H), (20,A.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 65, 242), 3, cv2.LINE_AA)
    # finally we display the image on a new window
    cv2.imshow('hand detection', A)


def closed_finger(points):
    ''' Function to check if the finger is closed or not, according to the position
    of the detected keypoints. If their position is relatively close to the reference
    keypoint 0 (calculated as euclidean distance), then the finger is closed, otherwise is open.
    Returns a map of finger closed, where in position 0 there is the thumb and in position 4 the pinkie.
    Handmap values are:
    0: CLOSED FINGER
    1: OPENED FINGER
    2: INDEX OPENED AND SUPERIMPOSED OVER THE OTHERS '''

    # empty handmap handle
    handmap = []

    j = 1
    # fingertips handle
    fop = []
    # for the 5 fingers (k) we read the finger kypoints (j)
    for k in range(0, 5):
        # empty finger handle
        finger = []
        # read finger keypoints, always 4 keypoints
        # i is the counter for points, j is the counter for the keypoint number,
        # which changes according to the current finger
        for i in range(j,j+4):
            # if the current point i exists, appends it
            if points[i]:
                finger.append(points[i])
            else:
                # else, appends keypoint 0
                # this is okay because the relative distance of each keypoint of the finger
                # calculated afterwards is relative to keypoint 0
                finger.append(points[0])

        # calculates relative distance of each keypoint of the finger relative to
        # keypoint 0. This is needed to find if the keypoints are collapsed or not
        distances = np.array([distance.euclidean(points[0], finger[0]),
                             distance.euclidean(points[0], finger[1]),
                             distance.euclidean(points[0], finger[2]),
                             distance.euclidean(points[0], finger[3])])
        # appends to fop handle the last value of distances, corresponding to
        # the fingertip distance from keypoint 0
        fop.append(distances[-1])

        # check if the current fingertip has a relative distance > 0 (not collapsed)
        if (distances[-1] > 0):
            # then, if any keypoint of the current finger is absent, set the whole finger as closed
            if np.any(distances==0):
                handmap.append(0)
            # calculates the proportional "length" of the finger as:
            # first value of distances (distance from the first keypoint
            # of the finger and keypoint 0) and the latest value of distances
            # (distance from the fingertip keypoint and keypoint 0), divided by
            # the fingerip distance. If this proportion is lower than 10%,
            # the finger is set as closed
            elif ((distances[-1] - distances[0])/distances[-1]) < 0.10:
                handmap.append(0)
            # if none of the above, the finger is open!
            else:
                handmap.append(1)
        # if the fingertip distance is not > 0 it means that the fingertip keypoint
        # is absent, thus we set the finger as closed
        else:
            handmap.append(0)

        # increment the keypoint values for the next finger
        j = j + 4

    # FOR ENDED! All fingers calculated by now!
    # check the fingertips distances: if the maximum distance in the list
    # corresponds to the index finger, then it means that the index was the only
    # finger open/detected as open, thus we impose the index value in the handmap as 2
    if max(fop) == fop[1]:
        handmap[1] = 2

    return handmap


def gesture(points, chain, acquire, chvalue, frame):
    ''' Function to check which gesture is performed. Right now only 2 gestures are defined:
    HAND OPEN: corresponds to handmap = [1,1,1,1,1], all fingers opened
    INDEX: corresponds to handmap = [_,2,_,_,_]
    NO GESTURE: if none of the above '''

    # obtains the current handmap
    handmap = closed_finger(points)

    # if all values of handmap are equal to 1, then the gesture is HAND OPEN
    if handmap.count(1) == len(handmap):
        rospy.loginfo(color.BOLD + color.GREEN + 'HAND OPEN' + color.END)
        G = 'HAND OPEN'
        # sets the acquisition flag to True and empties the chain queue
        acquire = True
        chain = []
    # if the finger value is equal to 2, then the gesture is INDEX
    elif handmap[1] == 2:
        rospy.loginfo(color.BOLD + color.CYAN + 'INDEX' + color.END)
        G = 'INDEX'
        # if the acquire flag has been set to True, the length of the chain
        # is less than the defined chain value and the index finger keypoint exists,
        # then it appends the index finger keypoint coordinates to the chain
        if acquire == True and len(chain) < chvalue and points[8]:
            chain.append(points[8])
            # EXPERIMENT TO FIND ESK: UNCOMMENT THIS LINE
            #error(points[8], frame)
    else:
        # no gesture was detected
        # we accept the possible noises of the detection, thus we do not empty
        # the chain queue when no gesture is detected. A stronger constraint could be
        # to empty the chain in this case, and only accept values if the chain is not broken
        rospy.loginfo(color.BOLD + color.PURPLE + 'NO GESTURE' + color.END)
        G = 'NO GESTURE'

    return chain, acquire, G, str(handmap)


#def showW(points, frame):
    ''' Function that projects the hand keypoints from workspace H (pixels)
        to workspace W (pixels), for visualization purposes '''

    # our workspace H (X right, Y down) corresponds to workspace W (X up, y right)
    # without scale factors, so we just need to change the coordinates!
    # when we move our hand right we must see the point going up (and vice versa)
    # when we move our hand down, we must see the point going left (and vice versa)

    # if index in workspace allora draw
    # mappare coordinate al contrario

#######

def myhook():
    ''' Hook called upon exiting the program using Ctrl + C '''

    rospy.loginfo(color.BOLD + color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + color.END)

def main():
    ''' Main program of Hands-Free! This program does the following:

    STEP 1: INITIALIZATION
            Some initialization procedures are carried out when the node is started,
            such as the OpenPose network initialization, the loading of the calibration
            parameters saved in the corresponding YAML files, the initialization of the
            Kinect camera object etc.
    STEP 2: FRAME ACQUISITION
            Starts the Kinect device and acquires the undistorted frame (applies coorections)
            passes this frame to the caffe network to detect the keypoints.
    STEP 3: GESTURE RECOGNITION
            According to the defined gestures and the detected keypoints, determines
            which gesture is present in the frame. If the chain of detected index finger positions
            has been filled, calculates the mean value (to reduce noise).
    STEP 4: ROBOT MOVEMENT
            Performs calculations to find the corresponding point from image frame
            to workspace H and from workspace H to robot coordinates.
            Then moves the robot in that calculated point.
    STEP 5: VISUALIZATION
            If the draw flag has been selected, draws the acquired frame and
            the skeleton on top of the hand and some debug info useful for the user.
    '''

    ###### STEP 1: INITIALIZATION
    # node creation
    rospy.init_node('kinect_node')
    # caffe network initialization
    net = init_network()
    #refmaster = cv2.imread('master10mm.png')

    # loading of the calibration parameters of both camera and robot
    K, D, R, t, Rd, td = loadcalibcamera('camera_calibration.yaml')
    RRobot = loadcalibrobot('robot_workspace_calibration.yaml')

    # finds the coordinates of the reference point 0 in workspace H, used
    # to refer all the other points in the same workspace!
    reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), Rd, td, K, D)
    reference = reference.flatten()
    pt0 = (int(round(reference[0],2)), int(round(reference[1],2)))
    reference = np.array([[reference[0], reference[1], 1.0]])
    Ref = px2meters(reference, K, Rd, td)
    B3 = (319, 646)
    A3 = (1087, 647)
    A1 = (1097, 140)
    pt1 = px2meters(np.array([[float(B3[0]), float(B3[1]), 1.0]]), K, Rd, td)
    # new1 = (pt1 - Ref)
    pt2 = px2meters(np.array([[float(A3[0]), float(A3[1]), 1.0]]), K, Rd, td)
    # new2 = (pt2 - pt1)
    pt3 = px2meters(np.array([[float(A1[0]), float(A1[1]), 1.0]]), K, Rd, td)
    # new3 = (pt3 - pt2)
    # new4 = (Ref - pt3)
    # print('Distance B1-B3: ' + str(new1))
    # print('Distance B3-A3: ' + str(new2))
    # print('Distance A3-A1: ' + str(new3))
    # print('Distance A3-B1: ' + str(new4))

    # Kinect object creation
    kinect = Kinect(True, True, True, True, K, D)

    # program flags and parameters that should not be changed
    nPoints = 22
    chain = []
    acquire = False
    G = 'INIT'
    H = 'INIT'
    # change it to False if you do not want to draw the skeleton on the frames
    draw = True
    # PARAMETERS THAT MAY BE CHANGED BY THE USER!!
    # caffe network threshold: if higher, less keypoints would be accepted
    threshold = 0.2
    # chain value: if lower, less index finger positions are acquired before moving the robot,
    # meaning that the reaction time is faster but may be less accurate due to disturbances
    chvalue = 7
    # x value: this is the value that we set as fixed because we control the robot positioning
    # only on plane ZY, thus we fix the depth (in our case X) according to workspace W
    # distance from the robot (to avoid accidental collisions)
    x = 0.7
    TOSAVE = []

    while not rospy.is_shutdown():
        ###### STEP 2: FRAME ACQUISITION
        # starts the kinect object acquisition
        kinect.acquire()
        # calls the undistortion method: the frames are UNDISTORTED from now on,
        # so the calibration matrixes that must be used are Rd and td!!
        frame = kinect.RGBundistorted.copy()

        # uncomment this line if you prefer to use the normal image instead,
        # remember to comment the previous one and to use R and t instead of Rd and td
        #frame = kinect.color_new.copy()

        # detects the hand keypoints - this is the inference function
        points, inference_time = hand_keypoints(net, frame, threshold, nPoints)

        ###### STEP 3: GESTURE RECOGNITION
        # if all points are None or the reference keypoint is None, then prints no gestures
        if all(x == None for x in points[1:]) or points[0] == None:
            rospy.loginfo(color.BOLD + color.RED + 'NO GESTURE FOUND IN IMAGE' + color.END)
            draw = False
            G = 'NO GESTURE'
            H = 'NONE'
        # else, it finds the correct gesture based on the keypoints detected
        else:
            chain, acquire, G, H = gesture(points, chain, acquire, chvalue, frame)
            draw = True
            # if the chain queue has been filled, calculates the mean value of the
            # index finger coordinates, calculates the corresponding value in workspace
            # W and finally moves the robot using the corresponding coordinates
            if len(chain) == chvalue:
                # sets the acquisition flag to false, this is needed to correclty
                # wait for another hand open gesture to allow another point to be acquired
                acquire = False
                # zips x coordinates in one sublist and y coordinats in another sublist
                rechain = zip(*chain)
                # so now we can perform the mean easily
                mean = np.array([[sum(rechain[0])/len(rechain[0]), sum(rechain[1])/len(rechain[1]), 1.0]])
                rospy.loginfo(color.BOLD + color.YELLOW + 'CHAIN VALUE REACHED. MEAN IS: ' + str(mean[0][0]) + ', ' + str(mean[0][1]) + color.END)
                TOSAVE.append((mean[0][0], mean[0][1]))
                # empty the queue
                chain = []

                ###### STEP 4: ROBOT MOVEMENT

                # finds the mean point real world coordinates in workspace H from image coordinates
                Ph = px2meters(mean, K, Rd, td)
                # finds the coordinates of the calculated point with respect to reference point
                new = (Ph - Ref)
                rospy.loginfo(color.BOLD + color.YELLOW + 'CALCULATED COORDINATES IN H: ' + str(new) + color.END)
                # calculates robot coordinates from starting point new in reference system H
                # if workspace H and W differ, you need to calibrate them too (in our case this was not
                # necessary because the two had the same size but different orientation)
                Pr = H2R(new, RRobot, x)
                rospy.loginfo(color.BOLD + color.YELLOW + 'CALCULATED COORDINATES FOR ROBOT: ' + str(Pr) + color.END)

        ###### STEP 5: VISUALIZATION
        draw_skeleton(frame, points, draw, inference_time, G, H, pt0, B3, A3, A1)
        # commands needed to correctly visualize opencv images
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break

    # upon exiting, closes the kinect object and shuts down the ROS node
    LISTA = []
    B2 = (315, 390)
    B3 = (318, 633)
    B4 = (506, 267)
    B5 = (506, 513)
    O1 = (698, 141)
    O0 = (696, 391)
    O2 = (693, 636)
    A4 = (890, 268)
    A5 = (886, 516)
    A1 = (1086, 140)
    A2 = (1081, 393)
    A3 = (1075, 643)
    PP = [pt0, B2, B3, B4, B5, O1, O0, O2, A4, A5, A1, A2, A3]
    for p in range(0,len(PP)):
        pointsW(LISTA, PP[p], Ref, K, Rd, td, RRobot)
    print(LISTA)
    rospy.loginfo(color.BOLD + color.GREEN + str(TOSAVE) + color.END)
    kinect.stop()
    rospy.on_shutdown(myhook)

if __name__ == '__main__':
    main()
