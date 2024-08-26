import rospy
import argparse
import graphical_utils as gu
import numpy as np
import utils

from geometry_msgs.msg import PoseStamped

#python3.8 calibrate_robot.py './yaml/master_workspace.yaml' './yaml/robot_workspace_calibration.yaml' './yaml/calibration.yaml'


def calibrate(ws1, ws2):
    ''' Function to calibrate two workspaces using the same number of points
    which coordinates are expressed according to the first and second reference system.
    The result is the rototranslation matrix Rt used to convert the point from ws1 to ws2 coordinates.
    According to the number of points coordinates, the resulting R is shaped correctly.
    For example, if only (x,y) coordinates are passed, R is a (3x3), otherwise a (4x4).
    ATTENTION: be sure to order the points correctly, so that point1 is the first
    for both ws1 and ws2 lists, point2 is the second for both, etc.

    INPUTS:
    - ws1: list of points expressed as the first reference system coordinates
    - ws2: list of points expressed as the second reference system coordinates

    OUTPUTS:
    - R: rototranslation matrix containing rotation and translation parameters,
         plus a row of zeros and a 1 in the bottom-right corner
    '''

    # convert points from ws1 to numpy array
    A = np.asarray(ws1)
    # each point should be expressed as homogeneous coordinates, aka adding a 1 at the end
    A = np.append(A, np.ones((A.shape[0],1)), axis=1)

    # define how to swap coordinates according to reference
    # basically we create a rule to swap coordinates by making the second tuple of
    # ref equal to the first

    # b is the vector containing the ws2 coordinates
    # in this case coordinates do not need to be converted to homogeneous
    b = np.asarray(ws2)
    # ws2_ = np.asarray(ws2)
    # b = ws2_.copy()
    # for i in range(0, len(ref[0])):
    #     try:
    #         # the coordinate is found with the same sign
    #         idx = ref[1].index(ref[0][i])
    #         b[:,i] = ws2_[:,idx]
    #     except:
    #         # in this case the coordinate swaps sign
    #         idx = ref[1].index(-ref[0][i])
    #         b[:,i] = -ws2_[:,idx]

    # solve linear system x = A\b
    # the result is a vector x of values that must be placed in the correct spots
    # of the rototranslation matrix by hand, namely r11 r12 r13 r21 r22 r23 r13 r23 r33 t1 t2 t3
    x = np.linalg.lstsq(A,b,rcond=None)
    # x is now an array of shape (4,1) where only the first element contains the actual
    # parameters of Rt! So we need to convert the result (only first element) to a list
    x = x[0].flatten()
    x = x.tolist()

    # define rototranslation matrix using the values of x
    # it may be different according to the length of points (aka if the third coordinate
    # is not present the matrix only has r11 r12 r21 r22 t1 t2)
    # | r11 r12 r13 tx |
    # | r21 r22 r23 ty |
    # | r31 r32 r33 tz |
    # |  0   0   0   1 |
    #R = [[-x[1], x[0], -x[2]], [x[0], x[1], x[3]], [0.0, 0.0, 1.0]]

    # note that solving a linear system means that resulting coefficients are found first,
    # this is why all translation parameters (which are only scalars and not multiplied
    # for variables) are last in the resulting vector
    if len(x) > 6:
        R = [[x[0], x[1], x[2], x[9]], [x[3], x[4], x[5], x[10]], [x[6], x[7], x[8], x[11]], [0.0, 0.0, 0.0, 1.0]]
    else:
        R = [[x[0], x[1], x[4]], [x[2], x[3], x[5]], [0.0, 0.0, 1.0]]
    # convert R from list to numpy array
    R = np.asarray(R)

    return R

def getXYZpoint(msg):
    """
    Function that is called after a new message has been received
    from the selected robot's end-effector state topic. It's needed to
    interpret the robot ros message and return only the arm's xyz position
    (specified in the robot's reference system).

    INPUTS:
    - msg: full message coming from robot ros topic

    OUTPUT:
    - x,y,z: cartesian position of the robot's arm specified in the robot's reference system
    """

    print(gu.Color.BOLD + gu.Color.GREEN + '-- ACQUIRED POINT IS: --' + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + f'(X: {msg.pose.position.x}, Y: {msg.pose.position.y}, Z: {msg.pose.position.z})' + gu.Color.END)
    return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

def main(H_master_yaml, W_master_yaml, calibration_yaml):
    """
    Program needed to perform a calibration procedure between the robot's reference system and
    the user reference system. The program writes in a YAML file the acquired positions,
    guiding the user in the process, and performs the workspace calibration afterwards,
    saving the result in the same YAML file.

    Please note that if the two have different orientation (i.e. robot's is vertical, user's is horizontal)
    this means that the corresponding frames and coordinates may be different!

    INPUTS:
    - H_master_yaml: path to yaml file containing the number of robot positions and coordinates
                     with respect to reference system H (user workspace)
    - W_master_yaml: path to yaml file containing the number of robot positions and coordinates
                     with respect to reference system  W (in which the robot operates)
    - calibration_yaml: path to yaml file in which the calibration result will be saved
                        containing the user-defined markers, the corresponding robot coordinates
                        and the calibration matrix R used to convert between the two ref. systems

    OUTPUTS:
    - saves the calibration file (at path defined by calibration_yaml) containing the
      markers positions, the corresponding robot coordinates and the calibration matrix R
    """

    rospy.init_node('robot_workspace_calibration_node')
    posenode = '/ur_hardware_interface/tool_pose'

    # Load YAML files
    dict_H = utils.yaml2dict(H_master_yaml)
    dict_W = utils.yaml2dict(W_master_yaml)

    # Access the 'Pose' and 'Markers' from dict_H
    pose_H = dict_H['Pose']
    markers_H = dict_H['Markers']

    # Access the 'Master', 'Robot', and 'Calibration' from dict_W
    pose_W = dict_W['Pose']
    markers_W = dict_W['Markers']
    
    # only acquires a position when ENTER is pressed, allowing to move the robot in manual guidance
    points = []

    i = 0
    print(gu.Color.BOLD + gu.Color.CYAN + f'-- MOVE THE ROBOT TO POSITION {pose_H[i]} --' + gu.Color.END)
    input(gu.Color.BOLD + gu.Color.YELLOW + '-- PRESS ENTER TO START ACQUISITION AFTER YOU ARE DONE --' + gu.Color.END)
    while not rospy.is_shutdown() and i < len(pose_H):
        msg = rospy.wait_for_message(posenode, PoseStamped)
        x, y, z = getXYZpoint(msg)
        points.append([x, y, z])
        i += 1
        if i < len(pose_H):
            print(gu.Color.BOLD + gu.Color.CYAN + f'-- MOVE THE ROBOT TO POSITION {pose_H[i]} --' + gu.Color.END)
            input(gu.Color.BOLD + gu.Color.YELLOW + '-- PRESS ENTER TO START ACQUISITION AFTER YOU ARE DONE --' + gu.Color.END)
        else:
            break

    R_H2W = calibrate(markers_H, markers_W)
    R_W2R = calibrate(markers_W, points)

    dictionary = {'Robot': points, 'Master_W': markers_W, 'Master_H': markers_H, 'H2WCalibration': R_H2W.tolist(), 'W2RCalibration': R_W2R.tolist()}
    utils.dict2yaml(dictionary, calibration_yaml)

    rospy.signal_shutdown(gu.Color.BOLD + gu.Color.GREEN + '-- DONE! EXITING PROGRAM --' + gu.Color.END)
    rospy.on_shutdown(utils.myhook)

def args_preprocess():
    """ Function that parses the arguments passed by command line and sets them as variables for the main function. """

    parser = argparse.ArgumentParser()
    parser.add_argument('H_master_yaml', type=str, help='Specifies path to YAML file containing workspace H master coordinates and points names.')
    parser.add_argument('W_master_yaml', type=str, help='Specifies path to YAML file containing workspace W master coordinates and points names.')
    parser.add_argument('calibration_yaml', type=str, help='Specifies path to YAML file that should be saved containing the H2W and W2R calibrations.')

    args = parser.parse_args()
    main(args.H_master_yaml, args.W_master_yaml, args.calibration_yaml)

if __name__ == '__main__':
    args_preprocess()
