import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['GLOG_minloglevel'] = '2'

import rospy
import math
import utils
import graphical_utils as gu
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker


def generate_circle_waypoints(center, radius, num_points, orientation):
    """
    Generate waypoints for a circle in the YZ plane.

    Args:
        center (tuple): The center of the circle (x, y, z).
        radius (float): The radius of the circle.
        num_points (int): Number of waypoints to generate along the circle.
        orientation (list): The fixed orientation for all waypoints.

    Returns:
        list: A list of waypoints dictionaries with 'position' and 'orientation' keys.
    """
    waypoints = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        y = center[1] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        waypoints.append({
            'position': (center[0],y, z),
            'orientation': orientation
        })
    return waypoints

def move_action(robot, waypoints, linear_speed):
    rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + '-- MOVING... --' + gu.Color.END)
    robot.move2cartesian(waypoints=waypoints, linear_speed=linear_speed)
    rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + '-- DONE --' + gu.Color.END)

def main():
    rospy.init_node('robot_movement_node')
    robot = utils.Robot()

    robot.add_table_to_scene()
    robot.add_a2_sheet_to_scene()
    robot.set_home()

    # Center of the circle in the XZ plane
    center = (-0.3, 0, 0.45)
    radius = 0.1  # Radius of the circle
    num_points = 20  # Number of points along the circle

    # Definizione dei tre punti per formare un triangolo nel piano YZ
    robot_points = [
        [0.4, 0.1, 0.3],   # Punto 1: y=0.1, z=0.2
        [0.4, 0.05, 0.4],  # Punto intermedio
        [0.4, 0.0, 0.3],   # Punto intermedio
        [0.4, -0.05, 0.2], # Punto intermedio
        [0.4, -0.1, 0.2],  # Punto 2: y=-0.1, z=0.2
        [0.4, -0.05, 0.3],# Punto intermedio
        [0.4, 0.0, 0.4]    # Punto 3: y=0.0, z=0.1
    ]

    # Calcola il quaternione per mantenere il robot parallelo all'asse y
    q = quaternion_from_euler(0, -math.pi / 2, 0)
    orientation = [q[0], q[1], q[2], q[3]]
    linear_speed = 0.1  # Velocità lineare del robot

    rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'Press Enter to start movement' + gu.Color.END)
    input()  # Wait for user to press Enter

    # Simulate the trajectory in RViz
    rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + '-- SIMULATING TRAJECTORY... --' + gu.Color.END)
    #waypoints = []

    
    #for p in robot_points:
    #    waypoints.append({'position': tuple(p), 'orientation': orientation})
    waypoints = generate_circle_waypoints(center, radius, num_points, orientation)
    print(waypoints)
    robot.move2cartesian(waypoints=waypoints, linear_speed=linear_speed, simulate_only=True)

    robot.visualize_trajectory_as_line(waypoints)  # Visualize the trajectory as a line
    rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + '-- SIMULATION DONE. Press Enter to execute the movement --' + gu.Color.END)
    input()  # Wait for user to press Enter to execute the movement

    # Execute the trajectory
    move_action(robot, waypoints, linear_speed)
    robot.delete_trajectory_marker()
    #robot.move2cartesian(waypoints=waypoints, linear_speed=linear_speed)

    robot.set_home()

if __name__ == '__main__':
    main()
