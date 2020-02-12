import numpy as np
import cv2
import rospy
from intera_core_msgs.msg import EndpointState
from utils import *

def getXYZpoint(msg):
    ''' Function that is called after a new message has been received
    from the selected robot end effector state topic '''

    print(color.BOLD + color.GREEN + 'Acquired point is: ' + color.END)
    print('(X: ' + str(msg.pose.position.x) + ', Y: ' + str(msg.pose.position.y) + ', Z: ' + str(msg.pose.position.z) + ')')
    return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

def myhook():
    ''' ROS hook called upon exiting using ctrl+C, used to exit cleanly '''

    rospy.loginfo(color.BOLD + color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + color.END)

def main():
    ''' Program needed to perform the robot to workspace W calibration. Please note that
    workspace W in our case is vertical, thus corresponding to plane Z-Y of the robot.
    The program writes in a YAML file the acquired positions, guiding the user in the process,
    and performs the workspace calibration afterwards, saving the result in the same YAML file. '''

    # initialize ros node
    rospy.init_node('robot_workspace_calibration_node')
    # define the robot endpoint state topic, in our case is this one because
    # we used a SAWYER robot. IT IS DIFFERENT ACCORDING TO THE ROBOT!!!
    posenode = '/robot/limb/right/endpoint_state'

    # we only acquire a position when we press ENTER, allowing us to move
    # the robot in manual guidance and position the end effector inside the centering
    # tool we 3D printed for the vertical workspace calibration. Check out our paper
    # for more details about the procedure!
    points = []
    pose = ['A1', 'A2', 'A3', 'A4', 'A5', 'O1', 'O0', 'O2', 'B1', 'B2', 'B3', 'B4', 'B5']
    i = 0
    raw_input(color.BOLD + color.YELLOW + '-- Starting! Press ENTER to continue' + color.END)
    while not rospy.is_shutdown() and i <= len(pose):
        print(color.BOLD + color.CYAN + 'Move the robot to position ' + str(pose[i]) + color.END)
        # we use wait for message because is a bettere option than the classic Subscriber
        # in this case, since the topic is continuosly written by the robot
        msg = rospy.wait_for_message(posenode, EndpointState)
        X,Y,Z = getXYZpoint(msg)
        # append the acquired points to the whole matrix of marker-robot positions
        points.append([X, Y, Z])
        i = i + 1
        # using this, we stop and acquire a new point only after ENTER is pressed again
        raw_input(color.BOLD + color.YELLOW + 'Press any key to continue' + color.END)

    # we should have acquired all the points, in our case 13!
    # now we write them in the YAML file along the Master points,
    # which corresponds to the marker positions according to the reference system W
    # centered in the bottom-left marker B1 of the master

    # CHANGE THIS SINCE YOUR WORKSPACE IS PROBABLY DIFFERENT!
    markers = [[0.0, 70], [22.5, 70.0], [45.0, 70.0],
              [11.25, 52.5], [33.75, 52.5], [0.0, 35.0],
              [22.5, 35.0], [45.0, 35.0], [0.0, 0.0],
              [22.5, 0.0], [45.0, 0.0], [11.25, 17.5], [33.75, 17.5]]

    R = calibrateW2R(markers, points)

    robot = {'Robot' : points}
    master = {'Master' : markers}
    calibrated = {'Calibration' : R.tolist()}

    dictionary = [master, robot, calibrated]
    dict2yaml(dictionary, 'robot_workspace_calibration.yaml')

    rospy.signal_shutdown(color.BOLD + color.RED + 'Program ended!' + color.END)
    rospy.on_shutdown(myhook)

if __name__ == '__main__':
    main()
