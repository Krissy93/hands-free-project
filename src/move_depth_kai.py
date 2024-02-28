#! /usr/bin/env python
import time

from KaiSDK.WebSocketModule import WebSocketModule
from KaiSDK.DataTypes import KaiCapabilities
import KaiSDK.Events as Events

import rospy
import utils

action = 0  # 0 no action, 1 homing, 2 move depth
movement = 0  # 0 stay, 1 move up, 2 move down
Debug = False


def rpyEv(ev):
    ''' Event listener for RPY, used to catch up/down gestures setting the movement global variable '''

    global movement
    if ev.pitch < -25:
        movement = 1
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'UP' + utils.Color.END)
    elif ev.pitch > 25:
        movement = 2
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'DOWN' + utils.Color.END)
    else:
        movement = 0
        if Debug:
            rospy.loginfo(utils.Color.BOLD + utils.Color.CYAN + 'STAY' + utils.Color.END)


def fingerShortcutEv(ev):
    ''' Event listener for finger position, used to catch actions gestures setting the action global variable '''

    global Debug
    global action
    if ev.fingers == [True, True, True, True]:
        action = 1
    elif ev.fingers == [False, False, False, False]:
        action = 2
    else:
        action = 0
    if Debug:
        print("action: " + str(action))


def main():
    ''' Program used to move the robot increasing/decreasing distance from desk via KAI input '''

    # depth increment
    incr = utils.incr

    print(utils.Color.BOLD + utils.Color.CYAN + 'Initializing node... ' + utils.Color.END)
    # initialize ros node
    rospy.init_node("increase_depth_kai")

    # initialize robot
    robot = utils.Robot()

    # Use your module's ID and secret here
    moduleID = "moduleName"
    moduleSecret = "qwerty"

    # Create a WS module and connect to the SDK
    module = WebSocketModule()
    success = module.connect(moduleID, moduleSecret)

    if not success:
        print(utils.Color.BOLD + utils.Color.RED + 'Unable to authenticate with Kai SDK' + utils.Color.END)
        rospy.signal_shutdown("")
    else:
        print(utils.Color.BOLD + utils.Color.CYAN + 'KAI successfully connected' + utils.Color.END)

    # Set the default Kai to record fingers changes
    module.setCapabilities(module.DefaultKai, KaiCapabilities.FingerShortcutData)

    while not rospy.is_shutdown():
        # Register event listeners
        module.DefaultKai.register_event_listener(Events.FingerShortcutEvent, fingerShortcutEv)

        if action < 2:
            module.unsetCapabilities(module.DefaultKai, KaiCapabilities.PYRData)
            if action == 1:
                robot.move_to_cartesian(utils.home[0] - 0.05, utils.home[1], utils.home[2] - utils.z_offset, time=0.00001, steps=1, Debug=Debug)
                if Debug:
                    print(utils.Color.BOLD + utils.Color.GREEN + ' -- ROBOT HOME POSE REACHED -- \n' + utils.Color.END)

        if action == 2:
            module.setCapabilities(module.DefaultKai, KaiCapabilities.PYRData | KaiCapabilities.FingerShortcutData)
            module.DefaultKai.register_event_listener(Events.PYREvent, rpyEv)
            time.sleep(0.1)

            if movement == 1:
                robot.increase_x(-incr, Debug=Debug)
            elif movement == 2:
                robot.increase_x(incr, Debug=Debug)

    module.close()
    rospy.on_shutdown(utils.myhook)


if __name__ == '__main__':
    main()
