#! /usr/bin/env python

import rospy
from intera_core_msgs.msg import EndpointState
import utils


def getXYZpoint(msg):
    ''' Function that is called after a new message has been received
    from the selected robot's end-effector state topic '''

    print(utils.Color.BOLD + utils.Color.GREEN + 'Acquired point is: ' + utils.Color.END)
    print('(X: ' + str(msg.pose.position.x) + ', Y: ' + str(msg.pose.position.y) + ', Z: ' + str(msg.pose.position.z) + ')')
    return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z


def main():
    ''' Program needed to perform the robot to workspace W calibration.
    The program writes in a YAML file the acquired positions, guiding the user in the process,
    and performs the workspace calibration afterwards, saving the result in the same YAML file.

     Please note that workspace W in our case is vertical, thus corresponding to plane Z-Y of the robot.'''

    # initialize ros node
    rospy.init_node('robot_workspace_calibration_node')
    # define the robot endpoint state topic
    posenode = '/robot/limb/right/endpoint_state'

    # only acquires a position when ENTER is pressed, allowing to move the robot in manual guidance
    points = []
    pose = ['A1', 'A2', 'A3', 'A4', 'A5', 'O1', 'O0', 'O2', 'B1', 'B2', 'B3', 'B4', 'B5']
    i = 0
    print(utils.Color.BOLD + utils.Color.CYAN + 'Move the robot to position ' + str(pose[i]) + utils.Color.END)
    raw_input(utils.Color.BOLD + utils.Color.YELLOW + '-- Starting! Press ENTER to continue' + utils.Color.END)
    while not rospy.is_shutdown() and i <= len(pose):
        # use wait for message because is a better option than the classic Subscriber
        # in this case, since the topic is continuosly written by the robot
        msg = rospy.wait_for_message(posenode, EndpointState)
        x, y, z = getXYZpoint(msg)
        # append the acquired points to the whole matrix of marker-robot positions
        points.append([x, y, z])
        i = i + 1
        if i <= len(pose)-1:
            # stop and acquire a new point only after ENTER is pressed again
            print(utils.Color.BOLD + utils.Color.CYAN + 'Move the robot to position ' + str(pose[i]) + utils.Color.END)
            raw_input(utils.Color.BOLD + utils.Color.YELLOW + 'Press any key to continue' + utils.Color.END)
        else:
            break

    # write all the points in the YAML file along the Master points,
    # which corresponds to the marker positions according to the reference system W
    # centered in the bottom-left marker B1 of the master

    # position of markers in W
    markers = [[0.0, 70.0], [22.5, 70.0], [45.0, 70.0],
              [11.25, 52.5], [33.75, 52.5], [0.0, 35.0],
              [22.5, 35.0], [45.0, 35.0], [0.0, 0.0],
              [22.5, 0.0], [45.0, 0.0], [11.25, 17.5], [33.75, 17.5]]

    R = utils.calibrateW2R(markers, points)

    robot = {'Robot': points}
    master = {'Master': markers}
    calibrated = {'Calibration': R.tolist()}

    dictionary = [master, robot, calibrated]
    utils.dict2yaml(dictionary, '/home/fole/sawyer_ws/src/hands_free/src/yaml/robot_workspace_calibration.yaml')

    rospy.signal_shutdown(utils.Color.BOLD + utils.Color.RED + 'Program ended!' + utils.Color.END)
    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()
