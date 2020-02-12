#! /usr/bin/env python

import rospy
import argparse
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
import PyKDL
from tf_conversions import posemath
from intera_interface import Limb

def move2cartesian(position=None, orientation=None, relative_pose=None, in_tip_frame=False, joint_angles=[],
         tip_name='right_hand', linear_speed=0.6, linear_accel=0.6, rotational_speed=1.57,
         rotational_accel=1.57, timeout=None, neutral=False):
    """
    Move the robot arm to the specified configuration.
    Call using:
    $ rosrun intera_examples go_to_cartesian_pose.py  [arguments: see below]
    -p 0.4 -0.3 0.18 -o 0.0 1.0 0.0 0.0 -t right_hand
    --> Go to position: x=0.4, y=-0.3, z=0.18 meters
    --> with quaternion orientation (0, 1, 0, 0) and tip name right_hand
    --> The current position or orientation will be used if only one is provided.
    -q 0.0 -0.9 0.0 1.8 0.0 -0.9 0.0
    --> Go to joint angles: 0.0 -0.9 0.0 1.8 0.0 -0.9 0.0 using default settings
    --> If a Cartesian pose is not provided, Forward kinematics will be used
    --> If a Cartesian pose is provided, the joint angles will be used to bias the nullspace
    -R 0.01 0.02 0.03 0.1 0.2 0.3 -T
    -> Jog arm with Relative Pose (in tip frame)
    -> x=0.01, y=0.02, z=0.03 meters, roll=0.1, pitch=0.2, yaw=0.3 radians
    -> The fixed position and orientation paramters will be ignored if provided
    """

    try:
        #rospy.init_node('go_to_cartesian_pose_py')
        limb = Limb()

        traj_options = TrajectoryOptions()
        traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
        traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)

        wpt_opts = MotionWaypointOptions(max_linear_speed=linear_speed,
                                         max_linear_accel=linear_accel,
                                         max_rotational_speed=rotational_speed,
                                         max_rotational_accel=rotational_accel,
                                         max_joint_speed_ratio=1.0)
        waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

        joint_names = limb.joint_names()

        if joint_angles and len(joint_angles) != len(joint_names):
            rospy.logerr('len(joint_angles) does not match len(joint_names!)')
            return None

        if neutral == True:
            limb.move_to_neutral()
        else:
            if (position is None and orientation is None and relative_pose is None):
                if joint_angles:
                    # does Forward Kinematics
                    waypoint.set_joint_angles(joint_angles, tip_name, joint_names)
                else:
                    rospy.loginfo("No Cartesian pose or joint angles given. Using default")
                    waypoint.set_joint_angles(joint_angles=None, active_endpoint=tip_name)
            else:
                endpoint_state = limb.tip_state(tip_name)
                if endpoint_state is None:
                    rospy.logerr('Endpoint state not found with tip name %s', tip_name)
                    return None
                pose = endpoint_state.pose

                if relative_pose is not None:
                    if len(relative_pose) != 6:
                        rospy.logerr('Relative pose needs to have 6 elements (x,y,z,roll,pitch,yaw)')
                        return None
                    # create kdl frame from relative pose
                    rot = PyKDL.Rotation.RPY(relative_pose[3],
                                             relative_pose[4],
                                             relative_pose[5])
                    trans = PyKDL.Vector(relative_pose[0],
                                         relative_pose[1],
                                         relative_pose[2])
                    f2 = PyKDL.Frame(rot, trans)
                    # and convert the result back to a pose message
                    if in_tip_frame:
                      # end effector frame
                      pose = posemath.toMsg(posemath.fromMsg(pose) * f2)
                    else:
                      # base frame
                      pose = posemath.toMsg(f2 * posemath.fromMsg(pose))
                else:
                    if position is not None and len(position) == 3:
                        pose.position.x = position[0]
                        pose.position.y = position[1]
                        pose.position.z = position[2]
                    if orientation is not None and len(orientation) == 4:
                        pose.orientation.x = orientation[0]
                        pose.orientation.y = orientation[1]
                        pose.orientation.z = orientation[2]
                        pose.orientation.w = orientation[3]
                poseStamped = PoseStamped()
                poseStamped.pose = pose

                if not joint_angles:
                    # using current joint angles for nullspace bais if not provided
                    joint_angles = limb.joint_ordered_angles()
                    waypoint.set_cartesian_pose(poseStamped, tip_name, joint_angles)
                else:
                    waypoint.set_cartesian_pose(poseStamped, tip_name, joint_angles)

            rospy.loginfo('Sending waypoint: \n%s', waypoint.to_string())

            traj.append_waypoint(waypoint.to_msg())

            result = traj.send_trajectory(timeout=timeout)
            if result is None:
                rospy.logerr('Trajectory FAILED to send')
                return

            if result.result:
                rospy.loginfo('Motion controller successfully finished the trajectory!')
            else:
                rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                             result.errorId)
                             
    except rospy.ROSInterruptException:
        rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
