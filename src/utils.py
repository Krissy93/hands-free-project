#!/usr/bin/env python

import yaml
import numpy as np
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
import cv2
import rospy

class color():
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

class Kinect():
    ''' Kinect object, it uses pylibfreenect2 as interface to get the frames.
    The original example was taken from the pylibfreenect2 github repository, at:
    https://github.com/r9y9/pylibfreenect2/blob/master/examples/selective_streams.py  '''

    def __init__(self, enable_rgb, enable_depth, need_bigdepth, need_color_depth_map, K=None, D=None):
        ''' Init method called upon creation of Kinect object '''

        # according to the system, it loads the correct pipeline
        # and prints a log for the user
        try:
            from pylibfreenect2 import OpenGLPacketPipeline
            self.pipeline = OpenGLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenCLPacketPipeline
                self.pipeline = OpenCLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline
                self.pipeline = CpuPacketPipeline()

        rospy.loginfo(color.BOLD + color.YELLOW + '-- PACKET PIPELINE: ' + str(type(self.pipeline).__name__) + ' --' + color.END)

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.K = K
        self.D = D

        # creates the freenect2 device
        self.fn = Freenect2()
        # if no kinects are plugged in the system, it quits
        self.num_devices = self.fn.enumerateDevices()
        if self.num_devices == 0:
            rospy.loginfo(color.BOLD + color.RED + '-- ERROR: NO DEVICE CONNECTED!! --' + color.END)
            sys.exit(1)

        # otherwise it gets the first one available
        self.serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(self.serial, pipeline=self.pipeline)

        # defines the streams to be acquired according to what the user wants
        types = 0
        if self.enable_rgb:
            types |= FrameType.Color
        if self.enable_depth:
            types |= (FrameType.Ir | FrameType.Depth)
        self.listener = SyncMultiFrameListener(types)

        # Register listeners
        if self.enable_rgb:
            self.device.setColorFrameListener(self.listener)
        if self.enable_depth:
            self.device.setIrAndDepthFrameListener(self.listener)

        if self.enable_rgb and self.enable_depth:
            self.device.start()
        else:
            self.device.startStreams(rgb=self.enable_rgb, depth=self.enable_depth)

        # NOTE: must be called after device.start()
        if self.enable_depth:
            self.registration = Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

        # last number is bytes per pixel
        self.undistorted = Frame(512, 424, 4)
        self.registered = Frame(512, 424, 4)

        # Optinal parameters for registration
        self.need_bigdepth = need_bigdepth
        self.need_color_depth_map = need_color_depth_map

        if self.need_bigdepth:
            self.bigdepth = Frame(1920, 1082, 4)
        else:
            self.bigdepth = None

        if self.need_color_depth_map:
            self.color_depth_map = np.zeros((424, 512),  np.int32).ravel()
        else:
            self.color_depth_map = None


    def acquire(self):
        ''' Acquisition method to trigger the Kinect to acquire new frames. '''

        # acquires a frame only if it's new
        frames = self.listener.waitForNewFrame()

        if self.enable_rgb:
            self.color = frames["color"]
            self.color_new = cv2.resize(self.color.asarray(), (int(1920 / 1), int(1080 / 1)))
            # The image obtained has a fourth dimension which is the alpha value
            # thus we have to remove it and take only the first three
            self.color_new = self.color_new[:,:,0:3]
            # correct distortion of camera
            self.correct_distortion()
        if self.enable_depth:
            # these only have one dimension, we just need to convert them to arrays
            # if we want to perform detection on them
            self.depth = frames["depth"]
            self.depth_new = cv2.resize(self.depth.asarray() / 4500., (int(512 / 1), int(424 / 1)))

            self.registration.undistortDepth(self.depth, self.undistorted)
            self.undistorted_new = cv2.resize(self.undistorted.asarray(np.float32) / 4500., (int(512 / 1), int(424 / 1)))

            self.ir = frames["ir"]
            self.ir_new = cv2.resize(self.ir.asarray() / 65535., (int(512 / 1), int(424 / 1)))

        if self.enable_rgb and self.enable_depth:
            self.registration.apply(self.color, self.depth, self.undistorted, self.registered,
                                    bigdepth=self.bigdepth, color_depth_map=self.color_depth_map)
            # RGB + D
            self.registered_new = self.registered.asarray(np.uint8)

            if self.need_bigdepth:
                self.bigdepth_new = cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1)))
                #cv2.imshow("bigdepth", cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1))))
            if self.need_color_depth_map:
                #cv2.imshow("color_depth_map", self.color_depth_map.reshape(424, 512))
                self.color_depth_map_new = self.color_depth_map.reshape(424, 512)

        # do this anyway to release every acquired frame
        self.listener.release(frames)

    def stop(self):
        ''' Stop method to close device upon exiting the program '''

        rospy.loginfo(color.BOLD + color.RED + '\n -- CLOSING DEVICE... --' + color.END)
        self.device.stop()
        self.device.close()

    def correct_distortion(self):
        ''' Method to correct distortion using camera calibration parameters '''

        if self.K is not None and self.D is not None:
            h,  w = self.color_new.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K,self.D,(w,h),1,(w,h))
            # undistort
            self.RGBundistorted = cv2.undistort(self.color_new, self.K, self.D, None, newcameramtx)
            # crop the image
            x,y,w,h = roi
            self.RGBundistorted = self.RGBundistorted[y:y+h, x:x+w]
            self.RGBundistorted = cv2.flip(self.RGBundistorted, 0)
            self.RGBundistorted = self.RGBundistorted[0:900, 520:1650]
        else:
            rospy.loginfo(color.BOLD + color.RED + '-- ERROR: NO CALIBRATION LOADED!! --' + color.END)
            self.RGBundistorted = self.color_new

def yaml2dict(path):
    ''' Function needed to load a YAML file from folder
    and return the corresponding dictionary. '''

    with open(path, 'r') as file:
        rospy.loginfo(color.BOLD + color.YELLOW + 'Reading YAML file...' + color.END)
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        dictionary = yaml.load(file, Loader=yaml.FullLoader)

        return dictionary

def dict2yaml(dictionary, path):
    ''' Function needed to write a given dictionary to a YAML file
    The structure of the dictionary to properly write and read the YAML is like this:
    dict_file = [{'sports' : ['soccer', 'football', 'basketball']},
                 {'countries' : ['Pakistan', 'USA', 'India']}]
    this creates a YAML file like this:
    - sports
        - soccer
        - football
        - basketball
    - countries
        - Pakistan
        - USA
        - India
    '''

    with open(path, 'w') as file:
        # dump simply writes the dictionary in the YAML file
        # it is not an append but a new write
        result = yaml.dump(dictionary, file)

        rospy.loginfo(color.BOLD + color.GREEN + 'YAML file saved!' + color.END)

def loadcalibcamera(path):
    '''Function that loads the calibration YAML file and returns calibration matrixes as arrays.
    Check that the calibration from camera_calibrate.py has been saved like this:
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
    points of workspace W. The matrix is saved by the file robot_workspace_calibration.py '''

    dictionary = yaml2dict(path)
    RtRobot = dictionary[2]['Calibration']
    RtRobot = np.asarray(RtRobot)

    return RtRobot

def calibrateW2R(M=None, R=None, path=None):
    '''This function may:
    1) load the robot calibration YAML file if it has been saved from
    vertical_workspace_calibration.py as: dict = [{'Master' : [[x, y], [x, y]]}, {'Robot' : [[x, y], [x, y]]}]
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
        # and are already lists!
        Master = M
        Robot = R
    else:
        print(color.BOLD + color.RED + 'ERROR, WRONG ARGUMENTS PASSED' + color.END)
        return

    # To correctly calibrate the vertical plane ZY, we need to solve
    # the linear system x = A\b where x contains the components of matrix
    # Rt (rototranslation) to convert robot coordinates to ref system W

    # from the original Master matrix containing the markers positions from
    # reference system W 0, we build a new matrix A. For each row of the original
    # Master matrix we build 2 rows of A like this: [[Mx -My 1 0], [My Mx 0 1]]
    # We had 13 marker positions, thus Master is of shape (13,2) and A of shape (13*2,4)

    A = []
    # we divide the values to obtain meters because the Master coordinates have been
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

def error(point, frame):
    ''' Function needed to find out the positioning error Esk between the real
    position of the index finger and the estimated position obtained by OpenPose network.
    Using this function and moving the hand around in different points of workspace H,
    we save the corresponding image frame in a given folder, and save the corresponding
    index finger coordinates in a txt file. '''

    files = next(os.walk('saved/'))[2]
    n = len(files)
    cv2.imwrite('saved/img_' + str(n+1) + '.png', frame)

    with open('points.txt', 'a') as file_object:
    # append the image number and the point value
        file_object.write('img_' + str(n+1) + '\t' + str(point) + '\n')
