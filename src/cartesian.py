import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_commander import MoveGroupCommander
from tf.transformations import quaternion_from_euler
import utils

def move2cartesian(position=None, orientation=None, relative_pose=None, in_tip_frame=False,
                   joint_angles=None, tip_name='tool0', linear_speed=0.6, linear_accel=0.6,
                   rotational_speed=1.57, rotational_accel=1.57, timeout=None, neutral=False):
    """
    Moves the UR3 robot arm to the specified configuration.

    Args:
        position (tuple): Tuple containing x, y, z of cartesian pose, i.e. (0.4, -0.3, 0.18) meters.
        orientation (tuple): Tuple containing quaternions of orientation, usually set at (0.5, 0.5, 0.5, 0.5) rads.
        relative_pose (tuple): Used only if no position or orientation are given and the pose is specified in the relative frame of the end effector. Contains both xyz and roll pitch yaw coordinates of pose.
        in_tip_frame (bool): Boolean variable setting the tip frame or not (aka if the position is to be considered in the tip frame or in the absolute reference system of the robot).
        joint_angles (list): Array containing the joint angles of the cartesian pose to reach needed to perform forward kinematics. Usually set to an empty list if position+orientation or relative pose are given, since it's filled afterwards.
        tip_name (str): Name of the end effector. For UR3, this can be 'tool0'.
        linear_speed (float): Sets the maximum linear speed for the trajectory planner (m/s).
        linear_accel (float): Sets the maximum linear acceleration for the trajectory planner (m/s^2).
        rotational_speed (float): Sets the maximum rotational speed for the trajectory planner (rad/s).
        rotational_accel (float): Sets the maximum rotational acceleration for the trajectory planner (rad/s^2).
        timeout (float): Time to wait after sending the waypoint. If set to None, waits forever.
        neutral (bool): Boolean setting if the robot should move to neutral pose or not.
    """
    try:
        # Initialize the move group commander for the UR3 arm
        moveit_commander.roscpp_initialize([])
        robot = utils.Robot()
        group = MoveGroupCommander("manipulator")  # Name of the move group for UR3

        # Set the reference frame for pose targets
        group.set_pose_reference_frame("base_link")

        # Set the speed and acceleration
        group.set_max_velocity_scaling_factor(linear_speed)
        group.set_max_acceleration_scaling_factor(linear_accel)

        # Go to neutral pose
        if neutral:
            group.set_named_target("home")
            plan = group.go(wait=True)
            group.stop()
            group.clear_pose_targets()
            return

        # Set the target pose
        pose_target = PoseStamped()
        pose_target.header.frame_id = "base_link"

        if position:
            pose_target.pose.position.x = position[0]
            pose_target.pose.position.y = position[1]
            pose_target.pose.position.z = position[2]

        if orientation:
            pose_target.pose.orientation = Quaternion(*orientation)

        if relative_pose:
            if len(relative_pose) != 6:
                rospy.logerr('Relative pose needs 6 elements (x,y,z,roll,pitch,yaw)')
                return

            current_pose = group.get_current_pose(tip_name).pose
            relative_position = relative_pose[:3]
            relative_orientation = quaternion_from_euler(*relative_pose[3:])

            if in_tip_frame:
                # Transformation relative to end effector
                pose_target.pose.position.x = current_pose.position.x + relative_position[0]
                pose_target.pose.position.y = current_pose.position.y + relative_position[1]
                pose_target.pose.position.z = current_pose.position.z + relative_position[2]
                pose_target.pose.orientation = Quaternion(
                    current_pose.orientation.x + relative_orientation[0],
                    current_pose.orientation.y + relative_orientation[1],
                    current_pose.orientation.z + relative_orientation[2],
                    current_pose.orientation.w + relative_orientation[3]
                )
            else:
                # Transformation relative to base frame
                pose_target.pose.position.x = relative_position[0]
                pose_target.pose.position.y = relative_position[1]
                pose_target.pose.position.z = relative_position[2]
                pose_target.pose.orientation = Quaternion(*relative_orientation)

        group.set_pose_target(pose_target, tip_name)

        # Plan and execute the motion
        plan = group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

        rospy.loginfo('Motion controller successfully finished the trajectory!')

    except rospy.ROSInterruptException:
        rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')
    except Exception as e:
        rospy.logerr('An error occurred: %s', str(e))
