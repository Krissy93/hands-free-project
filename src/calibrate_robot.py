import rospy
import argparse
import graphical_utils as gu
import numpy as np
import utils
import tf2_ros

from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

def calibrateW2R(M=None, R=None, path=None):
    '''This function may:
    1) load the robot calibration YAML file if it has been saved from
    vertical_workspace_calibration.py as: dict = [{'Master' : [[x, y], [x, y]]}, {'Robot' : [[x, y], [x, y]]}]
    2) load the point lists of both Master M and Robot points R

    in both cases, the function uses the two point lists to obtain the rototranslation
    matrix between the robot and workspace W, which is returned as [[r11 r12 t1],[r21 r22 t2],[0.0 0.0 1.0]]'''

    if path is not None:
        # in this case the user gave a path to the function,
        # meaning that Master and Robot are saved in a YAML file
        dictionary = utils.yaml2dict(path)

        Master = dictionary['Pose']
        Robot = dictionary['Markers']
    elif M is not None and R is not None:
        # in this case, Master and Robot have been passed as arguments,
        # and are already lists!
        Master = M
        Robot = R
    else:
        print(gu.color.BOLD + gu.color.RED + 'ERROR, WRONG ARGUMENTS PASSED' + gu.color.END)
        return

    # To correctly calibrate the vertical plane ZY, we need to solve
    # the linear system x = A\b where x contains the components of matrix
    # Rt (rototranslation) to convert robot coordinates to ref system W

    # from the original Master matrix containing the markers positions from
    # reference system W 0, we build a new matrix A. For each row of the original
    # Master matrix we build 2 rows of A like this: [[Mx -My 1 0], [My Mx 0 1]]
    # We had 13 marker positions, thus Master is of shape (13,2) and A of shape (13*2,4)

    A = []
    # we divide the values to obtain meters because the Master coordinates have been
    # saved as centimeters; robot coordinates have been read from the encorder so
    # these are already expressed in meters. The minus sign is related to how we have
    # defined our workspaces H and W with respect to the robot! Point B1 of workspace W
    # is the reference point and, with respect to this point, the other markers are considered
    # with positive coordinates
    for i in range(0, len(Master)):
        row1 = [Master[i][0]/100.0, Master[i][1]/100.0, 1.0, 0.0]
        row2 = [-Master[i][1]/100.0, Master[i][0]/100.0, 0.0, 1.0]
        A.append(row1)
        A.append(row2)

    # convert A from list to numpy array
    A = np.asarray(A)

    # b is the vector containing the robot z-y coordinates, it has shape (13*2,1)
    # and it is built appending first zi and then yi like so: [z1, y1, z2, y2, ...]

    b = []
    for i in range(0, len(Robot)):
        b.append(Robot[i][1])
        b.append(Robot[i][2])

    # convert b from list to numpy array
    b = np.asarray(b)

    # solve linear system x = A\b
    x = np.linalg.lstsq(A,b,rcond=None)
    # x is now an array of shape (4,1)
    # convert result to list
    x = x[0].tolist()

    # define rototranslation matrix using the values of x!
    R = [[x[0], x[1], x[2]], [x[1], x[0], x[3]], [0.0, 0.0, 1.0]]
    # convert R from list to numpy array
    R = np.asarray(R)

    return R


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
    print(f"A shape: {A.shape}")  # Verifica le dimensioni di A

    # each point should be expressed as homogeneous coordinates, aka adding a 1 at the end
    A = np.append(A, np.ones((A.shape[0],1)), axis=1)
    print(f"A shape after appending ones: {A.shape}")

    # define how to swap coordinates according to reference
    # basically we create a rule to swap coordinates by making the second tuple of
    # ref equal to the first

    # b is the vector containing the ws2 coordinates
    # in this case coordinates do not need to be converted to homogeneous
    b = np.asarray(ws2)
    print(f"b shape: {b.shape}")

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

    rospy.loginfo(x)    

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
    if len(x) > 12:
        R = [[x[0], x[1], x[2], x[9]], [x[3], x[4], x[5], x[10]], [x[6], x[7], x[8], x[11]], [0.0, 0.0, 0.0, 1.0]]
    else:
        R = [[x[0], x[1], x[4]], [x[2], x[3], x[5]], [0.0, 0.0, 1.0]]
    # convert R from list to numpy array
    R = np.asarray(R)

    return R


def get_transform(listener, target_frame, source_frame):
    try:
        transform = listener.lookup_transform(target_frame, source_frame, rospy.Time(0))
        position = transform.transform.translation
        return position.x, position.y, position.z
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Error looking up transform: {e}")
        return None, None, None

def main(H_master_yaml, W_master_yaml, calibration_yaml):
    rospy.init_node('robot_workspace_calibration_node')

    # Crea un listener per le trasformazioni tf
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer)


    # Load YAML files
    dict_H = utils.yaml2dict(H_master_yaml)
    dict_W = utils.yaml2dict(W_master_yaml)

    # loads the two masters poses and markers
    pose_H = dict_H['Pose']
    markers_H = dict_H['Markers']
    pose_W = dict_W['Pose']
    markers_W = dict_W['Markers']

    points = []

    i = 0
    print(gu.Color.BOLD + gu.Color.CYAN + f'-- MOVE THE ROBOT TO POSITION {pose_H[i]} --' + gu.Color.END)
    input(gu.Color.BOLD + gu.Color.YELLOW + '-- PRESS ENTER TO START ACQUISITION AFTER YOU ARE DONE --' + gu.Color.END)
    while not rospy.is_shutdown() and i < len(pose_H):
        try:
            rospy.sleep(1.0)  # Attendi un momento per permettere al listener di aggiornarsi
            x, y, z = get_transform(tf_buffer, 'base_link', 'tool0')
            if x is not None and y is not None and z is not None:
                points.append([x, y, z])
                print(f"Acquired point: ({x}, {y}, {z})")
            else:
                print("Failed to acquire point.")
            i += 1
            if i < len(pose_H):
                print(gu.Color.BOLD + gu.Color.CYAN + f'-- MOVE THE ROBOT TO POSITION {pose_H[i]} --' + gu.Color.END)
                input(gu.Color.BOLD + gu.Color.YELLOW + '-- PRESS ENTER TO START ACQUISITION AFTER YOU ARE DONE --' + gu.Color.END)
            else:
                break
        except rospy.ROSException as e:
            print(f"Error waiting for message: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    #R_H2W = calibrate(markers_H, points)
    R_H2W = calibrateW2R(pose_W,points)
    R_W2R = calibrate(markers_W, points)

    dictionary = {'Robot': points, 'Master_W': markers_W, 'Master_H': markers_H, 'H2WCalibration': R_H2W.tolist(), 'W2RCalibration': R_W2R.tolist()}
    utils.dict2yaml(dictionary, calibration_yaml)
    dictW = {'Pose': pose_W, 'Markers': points}
    utils.dict2yaml(dictW, W_master_yaml)

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
