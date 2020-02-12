#! /usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from cartesian import *
# sawyer interface
import intera_interface
from utils import *

def choosepose(z):
    #point = msg
    p = raw_input(color.BOLD + color.YELLOW + 'Please select position from list: \n (1) A4 \n (2) A5 \n (3) O1 \n (4) O0 \n (5) O2 \n (6) B4 \n (7) B5 \n' + color.END)

    if p == '1':
        pose = (z, 0.120405903145, 0.478197482544)
    elif p == '2':
        pose = (z, -0.104596256277, 0.475805971467)
    elif p == '3':
        pose = (z, 0.235136020732,  0.305366141427)
    elif p == '4':
        pose = (z, 0.00975782109046, 0.302260886902)
    elif p == '5':
        pose = (z, -0.216105837676, 0.300404212918)
    elif p == '6':
        pose = (z, 0.124171220353, 0.128821210782)
    elif p == '7':
        pose = (z, -0.101265656096, 0.126592118591)
    else:
        pose = None

    if pose is not None:
        print(color.BOLD + color.GREEN + 'Position: ' + str(pose) + color.END)
    else:
        print(color.BOLD + color.RED + 'Wrong position!' + color.END)

    #print(p)
    return pose


def myhook():
    ''' ROS hook called upon exiting using ctrl+C, used to exit cleanly '''

    rospy.loginfo(color.BOLD + color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + color.END)


def main():
    ''' Pubblica le coordinate dell'indice in tempo reale in un topic. Quando necessario
    manda le coordinate di movimento al robot. Pubblicare a video i due workspace come
    acquisizione RGB in tempo reale '''

    #rospy.init_node('laser_node')
    #posenode = '/robot/limb/right/endpoint_state'

    z = 0.75
    raw_input(color.BOLD + color.YELLOW + '-- Starting! Press any key to continue --' + color.END)
    while not rospy.is_shutdown():
        #sub = rospy.Subscriber(posenode, EndpointState, callback, queue_size=1)
        pose = choosepose(z)
        if pose is not None:
            move2cartesian(position=pose, orientation=(0.5, 0.5, 0.5, 0.5))
        raw_input(color.BOLD + color.YELLOW + '-- Press any key to continue --' + color.END)


    rospy.on_shutdown(myhook)

if __name__ == '__main__':
    main()
