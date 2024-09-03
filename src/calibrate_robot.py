import rospy
import argparse
import graphical_utils as gu
import numpy as np
import utils
import tf2_ros

from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

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

    A = np.asarray(ws1)
    A = np.append(A, np.ones((A.shape[0],1)), axis=1)

    b = np.asarray(ws2)

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    x = x.flatten().tolist()

    if len(x) >= 12:
        R = [[x[0], x[1], x[2], x[9]], 
             [x[3], x[4], x[5], x[10]], 
             [x[6], x[7], x[8], x[11]], 
             [0.0, 0.0, 0.0, 1.0]]
    elif len(x) >= 9:
        R = [[x[0], x[1], x[4]], 
             [x[2], x[3], x[5]], 
             [0.0, 0.0, 1.0]]
    else:
        raise ValueError("Insufficient data to construct the transformation matrix")

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

    # Access the 'Pose' and 'Markers' from dict_H
    pose_H = dict_H['Pose']
    markers_H = dict_H['Markers']

    # Dict_W è una lista di dizionari, quindi estrai i dizionari
    dict_W_content = dict_W[0]
    robot_dict = dict_W[1]

    if 'Master' in dict_W_content:
        markers_W = dict_W_content['Master']
    else:
        raise KeyError("Key 'Master' not found in dict_W_content.")
    
    if 'Robot' in robot_dict:
        points = robot_dict['Robot']
    else:
        raise KeyError("Key 'Robot' not found in robot_dict.")
    
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
