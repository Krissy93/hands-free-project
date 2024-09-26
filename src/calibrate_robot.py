import rospy
import argparse
import graphical_utils as gu
import numpy as np
import utils
import tf2_ros

from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

def calcola_rototraslazione(M, R):
    '''This function may:
    1) load the robot calibration YAML file if it has been saved from
    vertical_workspace_calibration.py as: dict = [{'Master' : [[x, y], [x, y]]}, {'Robot' : [[x, y], [x, y]]}]
    2) load the point lists of both Master M and Robot points R

    in both cases the function uses the two point lists to obtain the rototranslation
    matrix between the robot and workspace W, which is returned as [[r11 r12 t1],[r21 r22 t2],[0.0 0.0 1.0]]'''


    # If M and R are passed as arguments
    Master = M
    Robot = R

    A = []
    for i in range(len(Master)):
        # Trasformiamo le coordinate di Master per adattarle al piano YZ
        row1 = [Master[i][0] , -Master[i][1] , 1.0, 0.0]  
        row2 = [Master[i][1] , Master[i][0] , 0.0, 1.0]  # Inversione delle coordinate
        A.append(row1)
        A.append(row2)

    # Convert A from list to numpy array
    A = np.asarray(A)

    # Build the vector b
    b = []
    for i in range(len(Robot)):
        # Modifica per il piano YZ (Z diventa Y, Y diventa -Z)
        b.append(Robot[i][1])  # Y del robot
        b.append(Robot[i][2])  # Z del robot

    # Convert b from list to numpy array
    b = np.asarray(b)

    # Solve linear system x = A\b
    x = np.linalg.lstsq(A, b, rcond=None)[0].tolist()


    # Define rototranslation matrix using the values of x
    R = [[x[0], -x[1], x[2]], 
         [x[1], x[0], x[3]], 
         [0.0, 0.0, 1.0]]  # Ultima riga

    # Convert R from list to numpy array
    R = np.asarray(R)

    return R

def calibrateW2R(M=None, R=None, path=None):
    '''This function may:
    1) load the robot calibration YAML file if it has been saved from
    vertical_workspace_calibration.py as: dict = [{'Master' : [[x, y], [x, y]]}, {'Robot' : [[x, y], [x, y]]}]
    2) load the point lists of both Master M and Robot points R

    in both cases the function uses the two point lists to obtain the rototranslation
    matrix between the robot and workspace W, which is returned as [[r11 r12 t1],[r21 r22 t2],[0.0 0.0 1.0]]'''

    if path is not None:
        # In this case, load from YAML file
        dictionary = utils.yaml2dict(path)
        Master = dictionary['Pose']
        Robot = dictionary['Markers']
    elif M is not None and R is not None:
        # If M and R are passed as arguments
        Master = M
        Robot = R
    else:
        print("ERROR, WRONG ARGUMENTS PASSED")
        return

    A = []
    for i in range(len(Master)):
        # Trasformiamo le coordinate di Master per adattarle al piano YZ
        row1 = [Master[i][0] , -Master[i][1] , 1.0, 0.0]  
        row2 = [Master[i][1] , Master[i][0] , 0.0, 1.0]  # Inversione delle coordinate
        A.append(row1)
        A.append(row2)

    # Convert A from list to numpy array
    A = np.asarray(A)

    # Build the vector b
    b = []
    for i in range(len(Robot)):
        # Modifica per il piano YZ (Z diventa Y, Y diventa -Z)
        b.append(Robot[i][1])  # Y del robot
        b.append(Robot[i][2])  # Z del robot

    # Convert b from list to numpy array
    b = np.asarray(b)

    # Solve linear system x = A\b
    x = np.linalg.lstsq(A, b, rcond=None)[0].tolist()


    # Define rototranslation matrix using the values of x
    R = [[x[0], x[1], x[2]], 
         [-x[1], x[0], x[3]], 
         [0.0, 0.0, 1.0]]  # Ultima riga

    # Convert R from list to numpy array
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


    R_H2W = calibrateW2R(pose_W,points)
    R_H2W_2 = calcola_rototraslazione(pose_W,points)

    dictionary = {'Robot': points, 'Master_W': markers_W, 'Master_H': markers_H, 'H2WCalibration': R_H2W.tolist(), 'H2W_2': R_H2W_2.tolist()}
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
