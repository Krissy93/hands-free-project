#!/usr/bin/env python

import rospy
import utils

# constants for marker in simulator panel
x = 0.75

poses = {
        'A1': [x, 0.225, 1.455],
        'A2': [x, 0.0, 1.455],
        'A3': [x, -0.225, 1.455],
        'A4': [x, 0.1125, 1.28],
        'A5': [x, -0.1125, 1.28],
        'O1': [x, 0.225, 1.105],
        'O0': [x, 0.0, 1.105],
        'O2': [x, -0.225, 1.105],
        'B1': [x, 0.225, 0.755],
        'B2': [x, 0.0, 0.755],
        'B3': [x, -0.225, 0.755],
        'B4': [x, 0.1125, 0.93],
        'B5': [x, -0.1125, 0.93],
     }

x_offset = -0.05  # gripper dimension
y_offset = 0
z_offset = -0.93  # offset between robot base and origin of simulator axis


def main():
    ''' Program used to move the robot in the simulator on the point selected from markers list.  '''

    print(utils.Color.BOLD + utils.Color.CYAN + 'Initializing node... ' + utils.Color.END)
    # initialize ros node
    rospy.init_node("calibrate_robot_acquisition")

    # MOVEMENT
    # initialize robot
    robot = utils.Robot()

    done = False

    print(utils.Color.BOLD + utils.Color.CYAN + 'Please select point:' + utils.Color.END)
    print(utils.Color.BOLD + utils.Color.CYAN + 'A1       A2      A3 ' + utils.Color.END)
    print(utils.Color.BOLD + utils.Color.CYAN + '    A4       A5       ' + utils.Color.END)
    print(utils.Color.BOLD + utils.Color.CYAN + 'O1       O0      O2 ' + utils.Color.END)
    print(utils.Color.BOLD + utils.Color.CYAN + '    B4       B5      ' + utils.Color.END)
    print(utils.Color.BOLD + utils.Color.CYAN + 'B1       B2      B3 ' + utils.Color.END)

    while not done and not rospy.is_shutdown():

        # takes the string corresponding to the point
        point_in = raw_input()
        point_in = point_in.upper()
        point_in = point_in.replace(" ", "")

        # move the robot to the selected point or it prints an error indicating wrong point
        if point_in in poses:
            p = poses[point_in]
            robot.move_to_cartesian(p[0] + x_offset, p[1] + y_offset, p[2] + z_offset)
        else:
            print(utils.Color.BOLD + utils.Color.RED + 'ERROR: Invalid point' + utils.Color.END)

    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()
