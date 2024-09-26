import numpy as np
import rospy
import utils
import graphical_utils as gu

def px2meters(pt, K, R, t):
    ''' Conversion function from pixels to meters used to obtain the match between
    a point in pixel coordinates in the image frame and its corresponding position in
    the real world in meters. The returned point XYZ is in meters!

    INPUTS:
    - pt: point to convert, it's an array of [x, y, 1] homogeneous coordinates
    - K: camera matrix resulting from calibration
    - R: rotation matrix resulting from calibration
    - t: translation vector resulting from calibration

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
    #rospy.loginfo(f"XYZ: {XYZ}")

    return XYZ

def H2R_old(original_point, R_H2W, R_W2R, depth, debug=False):

    ''' Function to properly convert a given point (in meters) from workspace
    H to workspace W. The obtained robot position is used to move the robot to that point.
    Please note that moving a robot in cartesian coordinates could lead to interpolation
    errors depending on the point and on the robot itself. It is also a good practice to
    move the robot in its neutral position/home position at the startup of the program.

    INPUTS:
    - original_point: point (m) received from user workspace and corresponding to index finger coordinates
    - R_H2W: rototranslation matrix used to transform coordinates from workspace H to W
    - R_W2R: rototranslation matrix used to transform coordinates from workspace W to R
    - depth: index position only commands two coordinates, so depth is not
             computed. Instead, this coordinate is passed as argument. It may
             be fixed or computed by external sensors. Depth is a list [idx, val],
             containing (1) a value 0-1-2 corresponding to which coordinate should be changed
             and (2) the depth value
    - debug: flag to activate debug logs

    OUTPUTS:
    - robot_point: xyz coordinates of the given point (m) transformed in the robot coordinate system.
                   These will be sent to the robot ROS node to form a waypoint and move it.
    '''

    # Convert R_H2W and R_W2R to numpy arrays if they are lists
    R_H2W = np.array(R_H2W)
    R_W2R = np.array(R_W2R)

    # flatten the given point and transform it in homogeneous coordinates
    # original point is the point in meters expressed in H ref system coordinates
    original_point = original_point.flatten()
    rospy.loginfo(f"Original points: {original_point}")
 
    # one is the unused one and the last one is the homogeneous one  
    original_point = np.array([original_point[1], original_point[0], 1.0])
    if debug:
        rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'W point is: ' + str(original_point) + gu.Color.END)

    #original_point = np.append(original_point, np.array([1]), axis=0)

    # converts the point to W coordinates using the convertion matrix H2W
    # please note that only two coordinates are meaningful
  
    point_H2W = R_H2W.dot(original_point.T)

    rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'H point is: ' + str(point_H2W) + gu.Color.END)

    # transform the point using the robot rototranslation matrix: (3x3) * (3x1)
    robot_point = R_W2R.dot(point_H2W.T)
    
    # depth[0] may be 0, 1 or 2 corresponding to x, y or z coordinates
    #robot_point[depth[0]] = depth[1]

    robot_point_finale = [depth[1], robot_point[0], robot_point[1]]

    if debug:
        rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'Robot point_finale: ' + str(robot_point_finale) + gu.Color.END)

    return robot_point_finale

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
    #rospy.loginfo(f"Original points: {original_point}")
    original_point = np.array([original_point[0], original_point[1], 1.0])
    #rospy.loginfo(f"Original points: {original_point}")
    

    robot_point = R_H2W.dot(original_point.T)
    #rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'Robot point: ' + str(robot_point) + gu.Color.END)
    robot_point_finale = [depth[1], robot_point[0], robot_point[1]]
    #rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'Robot point_finale: ' + str(robot_point_finale) + gu.Color.END)

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
        #rospy.loginfo(p)
        # converts the point from pixels to meters
        p = np.array(p)
        point = px2meters(p, K, R, t)
        # finds the coordinates of the calculated point with respect to reference point
        point = point - ref_pt

        # calculates robot coordinates from starting point in reference system Hs
        # if workspace H and W differ, you need to calibrate them too
        rospy.loginfo(f"points: {point}")
        robot_points.append(H2R(point, R_H2W, depth))

    return robot_points