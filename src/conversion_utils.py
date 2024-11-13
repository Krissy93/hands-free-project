import numpy as np
import rospy
import utils
import graphical_utils as gu

def px2meters(pt, K, R, t, scale_factor = 1.0):
    ''' Conversion function from pixels to meters used to obtain the match between
    a point in pixel coordinates in the image frame and its corresponding position in
    the real world in meters. The returned point XYZ is in meters!

    INPUTS:
    - pt: point to convert, it's an array of [x, y, 1] homogeneous coordinates
    - K: camera matrix resulting from calibration
    - R: rotation matrix resulting from calibration
    - t: translation vector resulting from calibration
    - scale-factor: proportional factor to conver the points

    OUTPUTS:
    - XYZ: converted point in meters, it's an array
    '''

    # find the inverse matrix K^-1
    K2 = np.linalg.inv(K)
    # find the inverse matrix R^-1
    R2 = np.linalg.inv(R)
    # transpose initial point. Be sure to pass it as [Xpx, Ypx, 1.0]
    pt = pt.T
    # STEP 1: K^-1 * point -> (3x3) * (3x1) = 3x1
    S = K2.dot(pt)
    #rospy.loginfo(f"S: {S}")
    # STEP 2: (K^-1 * point) - t -> (3x1) - (3x1) = 3x1
    N = S - t
    #rospy.loginfo(f"N: {N}")
    # STEP 3: R^-1 * ((K^-1 * point) - t) -> (3x3) * (3x1) = (3x1)
    XYZ = R2.dot(N)

    return XYZ*scale_factor


def H2R(original_point, R_H2W, depth):
    ''' Function to properly convert a given point (in meters) from workspace
    H to workspace W. The obtained robot position is used to move the robot in that point.
    Please note that moving a robot in cartesian coordinates could lead to interpolation
    errors depending on the point and on the robot itself. It is also a good practice to
    move the robot in its neutral position/home position at the startup of the program. '''

    R_H2W = np.array(R_H2W)
    # flatten the given point and transform it in homogeneous coordinates
    # since we use place ZY instead of XY we must give to the function the Y first and the X second!
    original_point = original_point.flatten()

    original_point = np.array([original_point[0]*10, original_point[1]*10, 1.0])

    robot_point = R_H2W.dot(original_point.T)

    robot_point_finale = [depth[1], robot_point[0], robot_point[1]]

    return robot_point_finale

def px2R(points_list, K, R, t, R_H2W, depth, ref_pt, debug=False):
    ''' Function to convert a list of pixel points to the corresponding
    robot workspace's points (meters).

    INPUTS:
    - points_list: list of points to convert. This is usually a list if a trajectory
                   is sent to the robot, otherwise is one element only. Points are
                   saved as tuple (x,y,z)
    - K: camera matrix needed to convert the point from pixels to meters
    - R: rotation matrix needed to convert the point from pixels to meters
    - t: translation vector needed to convert the point from pixels to meters
    - R_H2W: rototranslation matrix needed to convert from workspace H to workspace W
    - R_W2R: rototranslation matrix needed to convert from workspace W to robot coordinates R
    - depth: list containing the coordinate to change and the value of depth. This is needed
             because depth (aka proximity between W and end-effector) is not computed accurately
             so it may be passed as a fixed value or as a value determined from other sensors.
             The list is [idx, val], where idx may be 0-1-2 corresponding to x,y,z coordinate
             and val is the actual value of depth
    - ref_pt: reference point to correctly convert pixel points. This is basically the (0,0) point
              of reference system H since the (0,0) of the image plane is different
    - debug: boolean flag to activate debugging info

    OUTPUTS:
    - robot_points: list containing the original pixel points converted in robot coordinates.
                    Please note that each point is an array!
    '''

    robot_points = []
    for p in points_list:
        # converts the point from pixels to meters 
        p = np.array(p, dtype=np.float64)    
        
        point = px2meters(p, K, R, t)
        # finds the coordinates of the calculated point with respect to reference point
        point = point - ref_pt

        # calculates robot coordinates from starting point in reference system Hs
        # if workspace H and W differ, you need to calibrate them too
        robot_points.append(H2R(point, R_H2W, depth))

    return robot_points