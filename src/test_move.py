#!/usr/bin/env python

import rospy
import graphical_utils as gu
import utils

# Posizioni predefinite

x = -0.4
x_off = 0.15  # Offset per l'end effector lungo l'asse X

positions = {
    '1': [x + x_off, 0.2275, 0.3],  
    '2': [x + x_off, 0.0, 0.3],  
    '3': [x + x_off, -0.2275, 0.3],  
    '4': [x + x_off, 0.2275, 0.45],
    '5': [x + x_off, 0.0, 0.45],
    '6': [x + x_off, -0.2275, 0.45],
    '7': [x + x_off, 0.2275, 0.6],
    '8': [x + x_off, 0.0, 0.6],
    '9': [x + x_off, -0.2275, 0.6],
    '0': [x , 0.0, 0.24]
}

def main():
    rospy.init_node("move_to_predefined_pose")
    print(gu.Color.BOLD + gu.Color.CYAN + 'Initializing node... ' + gu.Color.END)

    robot = utils.Robot()
    robot.add_table_to_scene()
    robot.add_a2_sheet_to_scene()

    # Muovi alla posizione neutra
    robot.set_home()
    #robot .set_neutral()
    print(gu.Color.BOLD + gu.Color.CYAN + 'Moved to neutral position.' + gu.Color.END)
    
    while not rospy.is_shutdown():
        point_in = input("Enter the point (1, 2, 3, 4, 5, 6, 7, 8, 9, 0): ").strip().upper()
        if point_in in positions:
            p = positions[point_in]
            robot.move_to_cartesian(p[0], p[1], p[2])
            rospy.sleep(2)  # Wait a bit before allowing the next input
        else:
            print(gu.Color.BOLD + gu.Color.RED + 'ERROR: Invalid point. Please enter 1, 2, 3, 4, 5, 6, 7, 8, 9, 0.' + gu.Color.END)

if __name__ == '__main__':
    main()
