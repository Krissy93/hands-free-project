#! /usr/bin/env python

import yaml
import numpy as np
import cv2
import sys
import rospy
import math
import caffe
import time
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import intera_interface
from scipy.spatial import distance


class Color:
    ''' Class used to print colored info on terminal.
    First call "BOLD", a color (e. g. "YELLOW")
    and at the end of the print call "END". '''

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# crop points
x1 = 55
y1 = 80
x2 = 610
y2 = 460

# pedestal height
z_offset = 0.93

# home position YZ
y_home = 0.0
z_home = 1.105

# home point XYZ
home = [0.5, y_home, z_home]

# constants for depth movement
incr = 0.01


def crop(img):
    img = img[y1:y2, x1:x2]
    return img


class Camera:
    ''' Class that represents a Camera object '''

    def __init__(self, enable_rgb, K=None, D=None):
        ''' Init method called upon creation of Camera object '''

        self.enable_rgb = enable_rgb
        self.K = K
        self.D = D

        # if no camera are plugged in the system it quits, otherwise it gets the first one available

        # if there's an integrated webcam (0) first USB camera connected is 1, otherwise is 0
        self.cap = cv2.VideoCapture(1)
        if self.cap is None or not self.cap.isOpened():
            print(Color.BOLD + Color.RED + '-- ERROR: NO DEVICE CONNECTED!! --' + Color.END)
            sys.exit(1)

    def acquire(self, correct=False):
        ''' Acquisition method to trigger the Camera to acquire new frames. '''

        # acquires a frame
        ret, frame = self.cap.read()

        if not self.enable_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.color_new = cv2.resize(frame, (int(640 / 1), int(480 / 1)))

        # correct distortion of camera
        if correct:
            self.correct_distortion()

    def stop(self):
        ''' Stop method to close device upon exiting the program '''

        print(Color.BOLD + Color.CYAN + '\n -- CLOSING DEVICE... --' + Color.END)
        self.cap.release()
        cv2.destroyAllWindows()

    def grab(self):
        ''' Clear buffer to take new fresh frames '''

        self.cap.grab()

    def correct_distortion(self):
        ''' Method to correct distortion using camera calibration parameters '''

        if self.K is not None and self.D is not None:
            h,  w = self.color_new.shape[:2]

            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))

            # undistort
            self.RGBundistorted = cv2.undistort(self.color_new, self.K, self.D, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            self.RGBundistorted = self.RGBundistorted[y:y+h, x:x+w]
        else:
            print(Color.BOLD + Color.YELLOW + '-- NO CALIBRATION LOADED!! --' + Color.END)
            self.RGBundistorted = self.color_new


class Robot:
    ''' Class that represents a Robot object '''

    def __init__(self):
        ''' Init method called upon creation of Robot object '''

        rp = intera_interface.RobotParams()
        valid_limbs = rp.get_limb_names()
        if not valid_limbs:
            rp.log_message("Cannot detect any limb parameters on this robot. Exiting.", "ERROR")
            return

        print(Color.BOLD + Color.CYAN + 'Enabling robot... ' + Color.END)

        # enables robot
        rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        init_state = rs.state().enabled
        rs.enable()

        self.limb = intera_interface.Limb(valid_limbs[0])

        print(Color.BOLD + Color.GREEN + ' -- ROBOT READY -- ' + Color.END)

    def move_to_cartesian(self, x, y, z, time=4.0, steps=400.0, Debug=False):
        ''' Method to move the robot to a desired point. '''

        if Debug:
            print(Color.BOLD + Color.CYAN + 'Moving to selected point..' + Color.END)

        rate = rospy.Rate(1 / (time / steps))  # Defaults to 100Hz command rate

        current_pose = self.limb.endpoint_pose()
        delta = Pose()
        delta.position.x = (current_pose['position'].x - x) / steps
        delta.position.y = (current_pose['position'].y - y) / steps
        delta.position.z = (current_pose['position'].z - z) / steps

        for d in range(int(steps), -1, -1):
            if rospy.is_shutdown():
                return
            step = Pose()
            step.position.x = d * delta.position.x + x
            step.position.y = d * delta.position.y + y
            step.position.z = d * delta.position.z + z

            rpy = quaternion_from_euler(0, math.pi / 2, 0)  # end effector parallel to vertical master plane

            step.orientation.x = rpy[0]
            step.orientation.y = rpy[1]
            step.orientation.z = rpy[2]
            step.orientation.w = rpy[3]

            # inverse kinematic request
            joint_angles = self.limb.ik_request(step)
            if joint_angles:
                self.limb.set_joint_positions(joint_angles)
            else:
                if Debug:
                    print(Color.BOLD + Color.YELLOW + 'Invalid angle. Retrying...' + Color.END)

            rate.sleep()

        if Debug:
            print(Color.BOLD + Color.GREEN + '-- MOVEMENT COMPLETED --' + Color.END)

        rospy.sleep(0.1)

    def increase_x(self, x_incr, time=0.00001, Debug=False):
        ''' Method to increase/decrease the robot distance from desk. '''

        if Debug:
            if x_incr > 0:
                print(Color.BOLD + Color.CYAN + 'Increasing distance..' + Color.END)
            else:
                print(Color.BOLD + Color.CYAN + 'Decreasing distance..' + Color.END)

        rate = rospy.Rate(1 / time)
        current_pose = self.limb.endpoint_pose()
        final_pose = Pose()
        final_pose.position.x = current_pose['position'].x + x_incr
        final_pose.position.y = current_pose['position'].y
        final_pose.position.z = current_pose['position'].z

        rpy = quaternion_from_euler(0, math.pi / 2, 0)  # end effector parallel to vertical master plane
        final_pose.orientation.x = rpy[0]
        final_pose.orientation.y = rpy[1]
        final_pose.orientation.z = rpy[2]
        final_pose.orientation.w = rpy[3]

        if rospy.is_shutdown():
            return

        # inverse kinematic request
        joint_angles = self.limb.ik_request(final_pose)
        if joint_angles:
            self.limb.set_joint_positions(joint_angles)
        else:
            if Debug:
                print(Color.BOLD + Color.YELLOW + 'Unable to reach pose' + Color.END)

        rate.sleep()

        if Debug:
            print(Color.BOLD + Color.GREEN + '-- MOVEMENT COMPLETED --' + Color.END)

        rospy.sleep(0.1)


def yaml2dict(path):
    ''' Function needed to load a YAML file from folder
    and return the corresponding dictionary. '''

    with open(path, 'r') as file:
        print(Color.BOLD + Color.CYAN + 'Reading YAML file...' + Color.END)
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        dictionary = yaml.load(file, Loader=yaml.FullLoader)

        return dictionary


def dict2yaml(dictionary, path):
    ''' Function needed to write a given dictionary to a YAML file '''

    with open(path, 'w') as file:
        # dump simply writes the dictionary in the YAML file
        # it is not an append but a new write
        result = yaml.dump(dictionary, file)
        print(Color.BOLD + Color.GREEN + 'YAML file saved!' + Color.END)


def loadcalibcamera(path):
    '''Function that loads the calibration YAML file and returns calibration matrixes as arrays.

    Check that the calibration from calibrate_camera.py has been saved like this:
    dict = [{'K' : [[],[],[]]}, {'D' : [[]]}, {'R' : [[],[],[]]}, {'t' : [[]]}, {'Rd' : [[],[],[]]}, {'td' : [[]]}]'''

    dictionary = yaml2dict(path)
    K = dictionary[0]['K']
    K = np.asarray(K)
    D = dictionary[1]['D']
    D = np.asarray(D)
    R = dictionary[2]['R']
    R = np.asarray(R)
    t = dictionary[3]['t']
    t = np.asarray(t)
    Rd = dictionary[4]['Rd']
    Rd = np.asarray(Rd)
    td = dictionary[5]['td']
    td = np.asarray(td)

    return K, D, R, t, Rd, td


def loadcalibrobot(path):
    ''' Function that loads the Robot calibration YAML file and returns the complete
    rototranslation matrix Rt needed to properly move the robot in the corresponding
    points of workspace W. '''

    dictionary = yaml2dict(path)
    RtRobot = dictionary[2]['Calibration']
    RtRobot = np.asarray(RtRobot)

    return RtRobot


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

    return XYZ * 10  # change to meters


def calibrateW2R(M=None, R=None, path=None):
    ''' This function may:
    1) load the robot calibration YAML file if it has been saved from
    calibrate_robot.py as: dict = [{'Master' : [[x, y], [x, y]]}, {'Robot' : [[x, y], [x, y]]}]
    2) load the point lists of both Master M and Robot points R

    in both cases, the function uses the two point lists to obtain the rototranslation
    matrix between the robot and workspace W, which is returned as [[r11 r12 t1],[r21 r22 t2],[0.0 0.0 1.0]]'''

    if path is not None:
        # in this case the user gave a path to the function,
        # meaning that Master and Robot are saved in a YAML file
        dictionary = yaml2dict(path)

        Master = dictionary[0]['Master']
        Robot = dictionary[0]['Robot']
    elif M is not None and R is not None:
        # in this case, Master and Robot have been passed as arguments,
        # and are already lists
        Master = M
        Robot = R
    else:
        print(Color.BOLD + Color.RED + 'ERROR, WRONG ARGUMENTS PASSED' + Color.END)
        return

    # To correctly calibrate the vertical plane ZY, we need to solve
    # the linear system x = A\b where x contains the components of matrix
    # Rt (rototranslation) to convert robot coordinates to ref system W

    # from the original Master matrix containing the markers positions from
    # reference system W 0, we build a new matrix A. For each row of the original
    # Master matrix we build 2 rows of A like this: [[Mx -My 1 0], [My Mx 0 1]]
    # We had 13 marker positions, thus Master is of shape (13,2) and A of shape (13*2,4)

    A = []
    # divides the values to obtain meters because the Master coordinates have been
    # saved as centimeters; robot coordinates have been read from the encorder so
    # these are already expressed in meters. The minus sign is related to how we have
    # defined our workspaces H and W with respect to the robot! Point B1 of workspace W
    # is the reference point and, with respect to this point, the other markers are considered
    # with positive coordinates
    for i in range(0, len(Master)):
        row1 = [-Master[i][0]/100.0, -Master[i][1]/100.0, 1.0, 0.0]
        row2 = [Master[i][1]/100.0, Master[i][0]/100.0, 0.0, 1.0]
        A.append(row1)
        A.append(row2)

    # convert A from list to numpy array
    A = np.asarray(A)

    # b is the vector containing the robot z-y coordinates, it has shape (13*2,1)
    # and it is built appending first zi and then yi like so: [z1, y1, z2, y2, ...]

    b = []
    for i in range(0, len(Robot)):
        b.append(Robot[i][1])
        b.append(Robot[i][2])

    # convert b from list to numpy array
    b = np.asarray(b)

    # solve linear system x = A\b
    x = np.linalg.lstsq(A,b,rcond=None)
    # x is now an array of shape (4,1)
    # convert result to list
    x = x[0].tolist()

    # define rototranslation matrix using the values of x!
    R = [[-x[0], x[1], x[2]], [x[1], x[0], x[3]], [0.0, 0.0, 1.0]]
    # convert R from list to numpy array
    R = np.asarray(R)

    return R


def init_network(path_openpose):
    ''' Function to initialize the Caffe network parameters. '''

    rospy.loginfo(Color.BOLD + Color.YELLOW + 'INITIALIZING CAFFE NETWORK...' + Color.END)
    protoFile = path_openpose+"/hand/pose_deploy.prototxt"
    weightsFile = path_openpose+"/hand/pose_iter_102000.caffemodel"

    net = caffe.Net(protoFile, 1, weights=weightsFile)
    caffe.set_mode_gpu()

    return net


def hand_keypoints(net, frame, threshold, nPoints):
    ''' This function is used to perform inference in real time on each acquired frame.
    The Caffe network initialized at the startup of the program is passed to this function
    to perform the hand keypoint estimation. Using the given threshold, the predicted
    keypoints may be accepted or discarded. '''

    # calculate the total inference time as a difference between the starting time
    # of this function and the time corresponding to the output creation
    before = time.time()

    # values needed by the model to correctly detect the keypoints
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


def closed_finger(points):
    ''' Function to check if the finger is closed or not, according to the position
    of the detected keypoints. If their position is relatively close to the reference
    keypoint 0 (calculated as euclidean distance), then the finger is closed, otherwise is open.
    Returns a map of finger closed, where in position 0 there is the thumb and in position 4 the pinkie.

    Handmap values are:
    0: CLOSED FINGER
    1: OPENED FINGER '''

    handmap = []

    j = 1
    # for the 5 fingers (k) we read the finger kypoints (j)
    for k in range(0, 5):
        finger = []
        # read finger keypoints, always 4 keypoints
        # i is the counter for points, j is the counter for the keypoint number,
        # which changes according to the current finger
        for i in range(j, j+4):
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

        # check if the current fingertip has a relative distance > 0 (not collapsed)
        if distances[-1] > 0:
            # then, if any keypoint of the current finger is absent, set the whole finger as closed
            if np.any(distances == 0):
                handmap.append(0)
            # calculates the proportional "length" of the finger as:
            # first value of distances (distance from the first keypoint
            # of the finger and keypoint 0) and the latest value of distances
            # (distance from the fingertip keypoint and keypoint 0), divided by
            # the fingerip distance. If this proportion is lower than 10%,
            # the finger is set as closed
            elif ((distances[-1] - distances[0])/distances[-1]) < 0.10:
                handmap.append(0)
            # if none of the above, the finger is open
            else:
                handmap.append(1)
        # if the fingertip distance is not > 0 it means that the fingertip keypoint
        # is absent, thus we set the finger as closed
        else:
            handmap.append(0)

        # increment the keypoint values for the next finger
        j = j + 4

    return handmap


def draw_skeleton(frame, points, draw):
    ''' Function to draw the skeleton of one hand according to a pre-defined pose pair scheme to the frame.
    Does not return anything. '''

    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                  [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    # draw skeleton on frame if the draw flag has been set as True
    if draw:
        for pair in POSE_PAIRS:
            # pose pairs represent the lines connecting two keypoints, used to correclty draw the skeleton
            partA = pair[0]
            partB = pair[1]

            # if there is a point in both keypoints of the pair, draws the point and the connected line
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 1)
                cv2.circle(frame, points[partA], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


def draw_workspace(frame, pt0, draw, undistort):
    ''' Function to draw the points corresponding to the markers in the workspace H.
    Does not return anything. '''

    # markers in the ws (manually calculated)
    if undistort:
        B2 = (41, 194)
        B3 = (42, 338)
        B4 = (150, 124)
        B5 = (152, 267)
        O1 = (261, 51)
        O0 = (262, 193)
        O2 = (265, 338)
        A4 = (373, 121)
        A5 = (375, 265)
        A1 = (481, 48)
        A2 = (483, 189)
        A3 = (485, 334)
    else:
        B2 = (45, 193)
        B3 = (44, 342)
        B4 = (156, 121)
        B5 = (157, 268)
        O1 = (268, 49)
        O0 = (270, 191)
        O2 = (272, 339)
        A4 = (380, 118)
        A5 = (384, 264)
        A1 = (488, 46)
        A2 = (492, 188)
        A3 = (496, 334)

    # draw points on frame if the draw flag has been set as True
    if draw:
        cv2.circle(frame, pt0, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, A1, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, A2, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, A3, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, A4, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, A5, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, O1, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, O0, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, O2, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, B2, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, B3, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, B4, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, B5, 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)


def draw_gesture_info(frame, inference_time, G, H, draw):
    ''' Function to draw the gesture infos including the gesture catched and the inference time.
    Does not return anything. '''

    # draw info on frame if the draw flag has been set as True
    if draw:
        cv2.putText(frame, 'INFERENCE TIME: ' + str(inference_time) + ' SEC', (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 65, 242), 1, cv2.LINE_AA)
        cv2.putText(frame, str(G) + ' || HANDMAP: ' + str(H), (100, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 65, 242), 1, cv2.LINE_AA)


def myhook():
    ''' ROS hook called upon exiting using ctrl+C, used to exit cleanly '''

    rospy.loginfo(Color.BOLD + Color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + Color.END)












