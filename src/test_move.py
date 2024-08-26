#!/usr/bin/env python

import rospy
import graphical_utils as gu
import utils

# Posizioni predefinite

y = 0.4
positions = {
    '1': [0.2, y, 0.15],  # Primo punto da testare
    '2': [0.0, y, 0.15],  # Secondo punto da testare
    '3': [-0.2, y, 0.15],  # Terzo punto da testare
    '4': [0.2, y, 0.325],
    '5': [0.0, y, 0.325],
    '6': [-0.2, y, 0.325],
    '7': [0.2, y, 0.5],
    '8': [0.0, y, 0.5],
    '9': [-0.2, y, 0.5]
}

def main():
    rospy.init_node("move_to_predefined_pose")
    print(gu.Color.BOLD + gu.Color.CYAN + 'Initializing node... ' + gu.Color.END)

    robot = utils.Robot()

    # Muovi alla posizione neutra
    robot.set_neutral()
    print(gu.Color.BOLD + gu.Color.CYAN + 'Moved to neutral position.' + gu.Color.END)

    print(gu.Color.BOLD + gu.Color.CYAN + 'Enter O1, O2, or O3 to move to the corresponding point.' + gu.Color.END)
    
    while not rospy.is_shutdown():
        point_in = input("Enter the point (1, 2, 3, 4, 5, 6, 7, 8, 9): ").strip().upper()
        if point_in in positions:
            p = positions[point_in]
            robot.move_to_cartesian(p[0], p[1], p[2])
            rospy.sleep(2)  # Wait a bit before allowing the next input
        else:
            print(gu.Color.BOLD + gu.Color.RED + 'ERROR: Invalid point. Please enter 1, 2, 3, 4, 5, 6, 7, 8, 9.' + gu.Color.END)

if __name__ == '__main__':
    main()
