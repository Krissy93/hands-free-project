import rospy
import argparse
import graphical_utils as gu
import numpy as np
import utils
import tf2_ros

from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


import numpy as np

def calibrateW2R_test(pose, Robot_points):
    """
    Calcola la matrice di rototraslazione tra le pose del workspace W (in cm) e i punti del robot acquisiti (in metri).
    
    Parametri:
    - pose: Lista di pose dei marker nel workspace W (coordinate xy in cm).
    - Robot_points: Lista di punti acquisiti dal robot (coordinate xyz in metri).
    
    Restituisce:
    - Matrice di rototraslazione R 3x3.
    """
    # Convertiamo i marker (pose) da cm a metri
    pose_W_meters = np.array(pose) / 100.0  # Conversione da cm a metri
    
    # Estraiamo solo le coordinate y e z dai punti del robot
    robot_coords = np.array(Robot_points)[:, 1:]  # Considera solo le coordinate yz

    # Trasformiamo le pose W per adattarle al piano YZ
    # X diventa Y e Y diventa -Z
    transformed_pose_W = np.array([[pose[0], -pose[1]] for pose in pose_W_meters])  # Cambiamo il segno della coordinata Y

    # Centroide delle coordinate
    centroid_marker = np.mean(transformed_pose_W, axis=0)
    centroid_robot = np.mean(robot_coords, axis=0)
    
    # Sottraggo il centroide per centrare i dati
    centered_markers = transformed_pose_W - centroid_marker
    centered_robots = robot_coords - centroid_robot
    
    # Calcolo la matrice di covarianza
    H = np.dot(centered_markers.T, centered_robots)
    
    # Decomposizione SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Calcolo la matrice di rotazione
    R_rot = np.dot(Vt.T, U.T)
    
    # Correzione nel caso in cui la matrice di rotazione abbia una riflessione (determinante = -1)
    if np.linalg.det(R_rot) < 0:
        Vt[1, :] *= -1  # Inverti l'asse y se necessario
        R_rot = np.dot(Vt.T, U.T)

    # Creiamo la matrice di rototraslazione 3x3
    R = np.zeros((3, 3))
    R[0, 0] = R_rot[0, 0]  # R[0,0] = r11
    R[0, 1] = R_rot[0, 1]  # R[0,1] = r12
    R[1, 0] = R_rot[1, 0]  # R[1,0] = r21
    R[1, 1] = R_rot[1, 1]  # R[1,1] = r22
    R[2, 0] = 0.0          # R[2,0] = 0
    R[2, 1] = 0.0          # R[2,1] = 0
    R[2, 2] = 1.0          # R[2,2] = 1

    # Aggiungiamo la traslazione (centroide robot) per la matrice di rototraslazione
    R[0, 2] = centroid_robot[0]  # Traslazione X
    R[1, 2] = centroid_robot[1]  # Traslazione Y

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

    # Print the Master and Robot data
    print("Master (M):", Master)
    print("Robot (R):", Robot)

    A = []
    for i in range(len(Master)):
        # Trasformiamo le coordinate di Master per adattarle al piano YZ
        row1 = [Master[i][0] / 100.0, -Master[i][1] / 100.0, 1.0, 0.0]  # X diventa Y, Y diventa -Z
        row2 = [-Master[i][1] / 100.0, Master[i][0] / 100.0, 0.0, 1.0]  # Inversione delle coordinate
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

    rospy.loginfo(x)

    # Define rototranslation matrix using the values of x
    R = [[x[0], x[1], x[2]], 
         [x[1], x[0], x[3]], 
         [0.0, 0.0, 1.0]]  # Ultima riga

    # Convert R from list to numpy array
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

    print(pose_W)
    print(points)


    #R_H2W = calibrate(markers_H, points)
    R_H2W = calibrateW2R(pose_W,points)
    R_H2W_test = calibrateW2R_test(pose_W,points)
    R_W2R = calibrate(markers_W, points)

    print(R_H2W_test)

    dictionary = {'Robot': points, 'Master_W': markers_W, 'Master_H': markers_H, 'H2WCalibration': R_H2W.tolist(), 'W2RCalibration': R_W2R.tolist(), 'H2W': R_H2W_test.tolist()}
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
