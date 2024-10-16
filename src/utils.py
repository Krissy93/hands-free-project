#! /usr/bin/env python

import yaml
import numpy as np
import cv2
import sys
import graphical_utils as gu
import rospy
# import math
# import time
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from scipy.spatial import distance
#
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
from std_msgs.msg import String  # Import String message type for ROS communication
from ur_msgs.srv import SetIO
#import intera_interface

import actionlib
from moveit_commander import RobotCommander, PlanningSceneInterface
from moveit_commander import MoveGroupCommander
from moveit_commander import roscpp_initialize, roscpp_shutdown
from moveit_msgs.msg import RobotTrajectory
import moveit_commander
from geometry_msgs.msg import PoseStamped, Quaternion, Point
import geometry_msgs.msg
import visualization_msgs.msg as viz_msgs
from visualization_msgs.msg import Marker
import moveit_msgs.msg


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
                print(gu.Color.BOLD + gu.Color.RED + '-- ERROR: NO DEVICE CONNECTED TO PORT 0 AND 1! --' + gu.Color.END)
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

        print(gu.Color.BOLD + gu.Color.CYAN + '\n -- CLOSING DEVICE... --' + gu.Color.END)
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
            print(gu.Color.BOLD + gu.Color.YELLOW + '-- NO CALIBRATION LOADED!! --' + gu.Color.END)
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

        rospy.loginfo(gu.Color.BOLD + gu.Color.YELLOW + '-- PACKET PIPELINE: ' + str(type(self.pipeline).__name__) + ' --' + gu.Color.END)

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.K = K
        self.D = D

        # creates the freenect2 device
        self.fn = Freenect2()
        # if no kinects are plugged in the system, it quits
        self.num_devices = self.fn.enumerateDevices()
        if self.num_devices == 0:
            rospy.loginfo(gu.Color.BOLD + gu.Color.RED + '-- ERROR: NO DEVICE CONNECTED!! --' + gu.Color.END)
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
            #self.color_new = self.color.copy()
            self.color_new = cv2.resize(self.color.asarray(), (int(1920 / 1), int(1080 / 1)))
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

        rospy.loginfo(gu.Color.BOLD + gu.Color.RED + '\n -- CLOSING DEVICE... --' + gu.Color.END)
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
            #self.RGBundistorted = cv2.flip(self.RGBundistorted, 1)
            #self.RGBundistorted = self.RGBundistorted[0:900, 300:1800]
            #self.RGBundistorted = self.RGBundistorted[0:900, 520:1650]
        else:
            #rospy.loginfo(Color.BOLD + Color.RED + '-- NO UNDISTORTION APPLIED --' + Color.END)
            self.RGBundistorted = self.color_new
            #self.RGBundistorted = cv2.flip(self.RGBundistorted, 1)

class Robot:
    def __init__(self):
        # Initialize move group commander for UR3 arm
        moveit_commander.roscpp_initialize(sys.argv)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.robot = RobotCommander()
        self.group = MoveGroupCommander("manipulator")  # Name of the move group for UR3
        self.group.set_pose_reference_frame("base_link")
        
        # Initialize the robot and the scene
        self.scene = PlanningSceneInterface()
        
        # Create a publisher to visualize trajectories in RViz
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)
        
        # Allow some time for the scene to initialize
        rospy.sleep(2)
        
        # Set the planner
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(10)

    def add_a2_sheet_to_scene(self):
        """
        Add an A2 sheet to the planning scene.
        """
        # Define A2 sheet pose
        sheet_pose = PoseStamped()
        sheet_pose.header.frame_id = "base_link"  # Ensure this is the correct frame
        sheet_pose.pose.position.x = -0.5
        sheet_pose.pose.position.y = 0.0
        sheet_pose.pose.position.z = 0.4  # Adjust the Z position to place the sheet at desired height
        sheet_pose.pose.orientation.w = 0.0

            # Rotate the sheet so that it is flat, with the longer side (59.4 cm) parallel to the floor
        q = quaternion_from_euler(-math.pi/2, 0, math.pi/2)  # No rotation if it is already aligned correctly
        # If it needs to be aligned vertically or rotated, adjust the euler angles accordingly:
        # For example, if you need it to stand vertically, you might use quaternion_from_euler(-math.pi / 2, 0, 0)
        sheet_pose.pose.orientation.x = q[0]
        sheet_pose.pose.orientation.y = q[1]
        sheet_pose.pose.orientation.z = q[2]
        sheet_pose.pose.orientation.w = q[3]
        
        # A2 sheet dimensions
        sheet_size = (0.594, 0.42, 0.01)  # (x, y, z) dimensions

        # Add A2 sheet to the planning scene
        self.scene.add_box("a2_sheet", sheet_pose, sheet_size)
        
        # Allow some time for the scene to update
        rospy.sleep(2)

    def add_table_to_scene(self):
        """
        Add a table to the planning scene.
        """
        # Define table dimensions and pose
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_link"  # Ensure this is the correct frame
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = -0.12
        table_pose.pose.position.z = -0.44  # Place the table below the robot
        table_pose.pose.orientation.w = 1.0
        
        # Table dimensions
        table_size = (0.70, 0.60, 0.86)  # (x, y, z)

        # Add table to the planning scene
        self.scene.add_box("table", table_pose, table_size)
        
        # Allow some time for the scene to update
        rospy.sleep(2)

    def set_home(self):
        """
        Moves the UR3 robot arm to the home position with the specified joint angles:
        Base: 180°
        Shoulder: -90°
        Wrist1: -180°
        Wrist2: -90°
        Wrist3: 0°
        """
        try:
            # Definisci i valori degli angoli dei giunti in radianti
            joint_goal = self.group.get_current_joint_values()
            joint_goal[0] = math.radians(180)   # Base
            joint_goal[1] = math.radians(-90)   # Shoulder
            joint_goal[2] = math.radians(90)     # Elbow (rimane a 0)
            joint_goal[3] = math.radians(-180)  # Wrist 1
            joint_goal[4] = math.radians(-90)   # Wrist 2
            joint_goal[5] = math.radians(0)     # Wrist 3

            # Imposta la posizione target e muovi il robot
            self.group.go(joint_goal, wait=True)
            self.group.stop()  # Ferma il movimento
            self.group.clear_pose_targets()  # Svuota i target di posa

            rospy.loginfo('Moved to home position.')
        except rospy.ROSInterruptException:
            rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
        except Exception as e:
            rospy.logerr('An error occurred: %s', str(e))


    def set_neutral(self):
        """
        Moves the UR3 robot arm to the neutral (home) position.
        """
        try:
            # Go to neutral pose
            self.group.set_named_target("home")
            plan = self.group.go(wait=True)
            self.group.stop()
            self.group.clear_pose_targets()

            rospy.loginfo('Moved to neutral position.')
        except rospy.ROSInterruptException:
            rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
        except Exception as e:
            rospy.logerr('An error occurred: %s', str(e))

    def move_to_cartesian(self, x, y, z, Debug=False):
        # Method to move the robot to a desired point.

        try:
            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x = x
            target_pose.position.y = y
            target_pose.position.z = z

            # Set a fixed orientation (you can adjust this if needed)
            q = quaternion_from_euler(0, -math.pi / 2, 0)
            #q = quaternion_from_euler(-math.pi / 2, 0, 0)
            #q = quaternion_from_euler(0, -math.pi / 2, math.pi)
            target_pose.orientation.x = q[0]
            target_pose.orientation.y = q[1]
            target_pose.orientation.z = q[2]
            target_pose.orientation.w = q[3]

            self.group.set_pose_target(target_pose)
            plan = self.group.plan()

            if plan[1]:  # Check if a valid plan was found
                rospy.loginfo("Plan found, visualizing in RViz...")
                self.group.execute(plan[1], wait=True)
                self.group.stop()
                self.group.clear_pose_targets()
                rospy.loginfo('Moved to the specified position.')
            else:
                rospy.logwarn("No valid plan found for the desired pose.")
        except rospy.ROSInterruptException:
            rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
        except Exception as e:
            rospy.logerr('An error occurred: %s', str(e))

    def increase_x(self, x_incr, time=0.00001, Debug=False):
        # Method to increase/decrease the robot distance from desk.

        current_pose = self.group.get_current_pose().pose
        current_pose.position.x += x_incr

        self.group.set_pose_target(current_pose)
        self.group.go(wait=True)

        if Debug:
            rospy.loginfo('-- MOVEMENT COMPLETED --')

        rospy.sleep(0.1)


    def move2cartesian(self, waypoints, linear_speed=0.1, linear_accel=0.1, simulate_only=False):
        """
        Moves the UR3 robot arm through a list of waypoints and visualizes the planned trajectory in RViz.

        Args:
            waypoints (list): List of waypoints where each waypoint is a dictionary with 'position' (tuple) 
                            and 'orientation' (tuple) keys.
            linear_speed (float): Sets the maximum linear speed for the trajectory planner (m/s).
            linear_accel (float): Sets the maximum linear acceleration for the trajectory planner (m/s^2).
            simulate_only (bool): If True, only simulates the trajectory in RViz without executing it.
        """
        try:
            # Set the speed and acceleration
            self.group.set_max_velocity_scaling_factor(linear_speed)
            self.group.set_max_acceleration_scaling_factor(linear_accel)
            self.group.set_goal_tolerance(0.01)  # Aumenta la tolleranza sugli obiettivi

            # List to store waypoints
            waypoints_list = []

            for wp in waypoints:
                pose_target = PoseStamped()
                pose_target.header.frame_id = "base_link"

                if 'position' in wp:
                    pose_target.pose.position.x = wp['position'][0]
                    pose_target.pose.position.y = wp['position'][1]
                    pose_target.pose.position.z = wp['position'][2]

                if 'orientation' in wp:
                    pose_target.pose.orientation = Quaternion(*wp['orientation'])

                waypoints_list.append(pose_target.pose)
            
            #rospy.loginfo("Waypoints for Cartesian Path: %s", waypoints_list)

            # Plan the Cartesian path connecting the waypoints
            (plan, fraction) = self.group.compute_cartesian_path(
                waypoints_list,   # waypoints to follow
                0.05            # eef_step (smaller value for finer resolution)
                )

            rospy.loginfo("Path planning fraction: %.2f%%", fraction * 100)

            if fraction < 1.0:
                rospy.logwarn("Only able to compute %.2f%% of the path", fraction * 100)

            if simulate_only:
                # Visualize the plan in RViz
                display_trajectory = moveit_msgs.msg.DisplayTrajectory()
                display_trajectory.trajectory_start = self.robot.get_current_state()
                display_trajectory.trajectory.append(plan)
                self.display_trajectory_publisher.publish(display_trajectory)

                rospy.loginfo('Planned trajectory visualized in RViz. Execute manually from RViz.')
            else:
                # Execute the plan
                self.group.execute(plan, wait=True)

            self.group.stop()
            self.group.clear_pose_targets()

            rospy.loginfo('Motion controller successfully finished the trajectory!')

        except rospy.ROSInterruptException:
            rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
        except Exception as e:
            rospy.logerr('An error occurred: %s', str(e))
    
    
    def visualize_trajectory_as_line(self, waypoints):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Line width
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for wp in waypoints:
            p = Point()
            p.x = wp['position'][0]
            p.y = wp['position'][1]
            p.z = wp['position'][2]
            marker.points.append(p)

        rospy.sleep(1)  # Allow time for RViz to initialize
        self.marker_pub.publish(marker)
        rospy.loginfo('Trajectory line published to /visualization_marker')

    def delete_trajectory_marker(self):
        """
        Deletes the trajectory marker from RViz.
        """
        # Create a marker to delete the previous trajectory
        marker = viz_msgs.Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = viz_msgs.Marker.LINE_STRIP
        marker.action = viz_msgs.Marker.DELETE  # Set action to DELETE

        # Publish the delete marker
        marker_pub = rospy.Publisher('/visualization_marker', viz_msgs.Marker, queue_size=10)
        rospy.sleep(1)  # Allow time for RViz to process
        marker_pub.publish(marker)
        rospy.loginfo('Trajectory line deleted from /visualization_marker')



def myhook():
    moveit_commander.roscpp_shutdown()

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
            dictionary = yaml.load(file, Loader=yaml.FullLoader)
            return dictionary
    except FileNotFoundError:
        print(gu.Color.BOLD + gu.Color.RED + 'File not found. Please check the path: ' + str(path) + gu.Color.END)
    except yaml.YAMLError as exc:
        print(gu.Color.BOLD + gu.Color.RED + 'Error parsing YAML file: ' + str(exc) + gu.Color.END)
    except Exception as e:
        print(gu.Color.BOLD + gu.Color.RED + 'An unexpected error occurred: ' + str(e) + gu.Color.END)



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


