#! /usr/bin/env python

import yaml
import numpy as np
import cv2
import sys
import graphical_utils as gu
# import rospy
# import math
# import time
# from geometry_msgs.msg import Pose
# from tf.transformations import quaternion_from_euler
# from pylibfreenect2 import Freenect2, SyncMultiFrameListener
# from pylibfreenect2 import FrameType, Registration, Frame
# import intera_interface
# from scipy.spatial import distance
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


class Camera:
    ''' Class that represents a Camera object '''

    def __init__(self, enable_rgb, K=None, D=None):
        ''' Init method called upon creation of Camera object '''

        self.enable_rgb = enable_rgb
        self.K = K
        self.D = D

        # if no camera is plugged in the system it quits, otherwise it gets the first one available
        # WARNING: if there's an integrated webcam (0) first USB camera connected is 1, otherwise is 0

        self.cap = cv2.VideoCapture(0)
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            if self.cap is None or not self.cap.isOpened():
                print(Color.BOLD + Color.RED + '-- ERROR: NO DEVICE CONNECTED TO PORT 0 AND 1! --' + Color.END)
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

class Kinect:
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

        rospy.loginfo(Color.BOLD + Color.YELLOW + '-- PACKET PIPELINE: ' + str(type(self.pipeline).__name__) + ' --' + Color.END)

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.K = K
        self.D = D

        # creates the freenect2 device
        self.fn = Freenect2()
        # if no kinects are plugged in the system, it quits
        self.num_devices = self.fn.enumerateDevices()
        if self.num_devices == 0:
            rospy.loginfo(Color.BOLD + Color.RED + '-- ERROR: NO DEVICE CONNECTED!! --' + Color.END)
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


    def acquire(self, correct):
        ''' Acquisition method to trigger the Kinect to acquire new frames. '''

        # acquires a frame only if it's new
        frames = self.listener.waitForNewFrame()

        if self.enable_rgb:
            self.color = frames["color"]
            self.color_new = cv2.resize(self.color.asarray(), (int(1920 / 1.5), int(1080 / 1.5)))
            # The image obtained has a fourth dimension which is the alpha value
            # thus we have to remove it and take only the first three
            self.color_new = self.color_new[:,:,0:3]
            # correct distortion of camera
            self.correct_distortion(correct)
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

        rospy.loginfo(Color.BOLD + Color.RED + '\n -- CLOSING DEVICE... --' + Color.END)
        self.device.stop()
        self.device.close()

    def correct_distortion(self, correct):
        ''' Method to correct distortion using camera calibration parameters '''

        if self.K is not None and self.D is not None and correct is not False:
            h,  w = self.color_new.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K,self.D,(w,h),1,(w,h))
            # undistort
            self.RGBundistorted = cv2.undistort(self.color_new, self.K, self.D, None, newcameramtx)
            # crop the image
            x,y,w,h = roi
            self.RGBundistorted = self.RGBundistorted[y:y+h, x:x+w]
            self.RGBundistorted = cv2.flip(self.RGBundistorted, 0)
            #self.RGBundistorted = self.RGBundistorted[0:900, 300:1800]
            #self.RGBundistorted = self.RGBundistorted[0:900, 520:1650]
        else:
            #rospy.loginfo(Color.BOLD + Color.RED + '-- NO UNDISTORTION APPLIED --' + Color.END)
            self.RGBundistorted = self.color_new
            self.RGBundistorted = cv2.flip(self.RGBundistorted, 0)

class Robot:
    ''' Class that represents a Robot object '''

    def __init__(self,tip_name="right_hand"):
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
        # Inserito da verificare
        self._tip_name=tip_name

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
            joint_angles = self.limb.ik_request(step,self._tip_name)
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
    and return the corresponding dictionary.

    INPUTS:
    - path: full path to YAML file to read

    OUTPUTS:
    - dictionary: dictionary of the file contents
    '''

    try:
        with open(path, 'r') as file:
            print(gu.Color.BOLD + gu.Color.CYAN + 'Reading YAML file...' + gu.Color.END)
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            dictionary = yaml.load(file, Loader=yaml.FullLoader)

            return dictionary
    except:
        print(gu.Color.BOLD + gu.Color.RED + 'Wrong path to YAML file, please check. Path is: ' + str(path) + gu.Color.END)

def dict2yaml(dictionary, path):
    ''' Function needed to write a given dictionary to a YAML file.

    INPUTS:
    - dictionary: dictionary to write
    - path: full path to YAML file to save. If it doesn't exist, this function creates it

    OUTPUTS:
    - None
    '''

    with open(path, 'w') as file:
        # dump simply writes the dictionary in the YAML file
        # it is not an append but a new write
        result = yaml.dump(dictionary, file)
        print(gu.Color.BOLD + gu.Color.GREEN + 'YAML file saved!' + gu.Color.END)

def myhook():
    ''' ROS hook called upon exiting using ctrl+C, used to exit cleanly '''

    rospy.loginfo(gu.Color.BOLD + gu.Color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + gu.Color.END)
