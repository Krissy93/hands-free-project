#!/usr/bin/env python

import rospy
import utils

import intera_external_devices


def main():
    ''' Program used to move the robot increasing/decreasing distance from desk via keyboard input '''

    # debug value: set True for debug logs, False otherwise
    Debug = False

    # depth increment
    incr = utils.incr

    print(utils.Color.BOLD + utils.Color.CYAN + 'Initializing node... ' + utils.Color.END)
    # initialize ros node
    rospy.init_node("increase_depth_keyboard")

    # initialize robot
    robot = utils.Robot()

    done = False

    print(utils.Color.BOLD + utils.Color.GREEN + '\n Press : \n  - [w] decrease\n  - [s] increase\n  - [h] homing\n' + utils.Color.END)
    while not done and not rospy.is_shutdown():

        c = intera_external_devices.getch()
        if c:
            # catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("")

            elif c in ['w']:
                if Debug:
                    print(utils.Color.BOLD + utils.Color.PURPLE + 'W pressed' + utils.Color.END)
                robot.increase_x(-incr, Debug=Debug)
            elif c in ['s']:
                if Debug:
                    print(utils.Color.BOLD + utils.Color.PURPLE + 'S pressed' + utils.Color.END)
                robot.increase_x(incr, Debug=Debug)

            # move to home point (o0)
            elif c in ['h']:
                if Debug:
                    print(utils.Color.BOLD + utils.Color.PURPLE + 'H pressed' + utils.Color.END)
                    print(utils.Color.BOLD + utils.Color.CYAN + 'Moving to home point..' + utils.Color.END)
                robot.move_to_cartesian(utils.home[0] - 0.05, utils.home[1], utils.home[2] - utils.z_offset, time=0.00001, steps=1, Debug=Debug)

    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()
