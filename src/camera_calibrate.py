import numpy as np
import cv2
import glob
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from utils import *

def draw(img, corners, imgpts):
    ''' Function to draw the reference system origin on the image frame '''

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def findRT(img, mtx, dist):
    ''' Function to extract R and t corresponding to the fixed reference system
    placed on the user workspace. This is the reference system origin used to
    calculate and refer the real marker positions (e. g. the index finger positions)
    with respect to a known origin for the user, which is different than the image reference
    (in this case placed on the top-left corner of the image, corresponding to pixel coordinates 0,0) '''

    # converts frame to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # finds the chessboard corners according to the chessboard size
    # we have used a chessboard of 10-1 squares along x and 7-1 squares along y
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # if the chessboard has been found, calculates R and t
    if ret == True:
        # define the criteria used to calculate precisely the chessboard corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points according to chessboard size
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # find out the precise positions of chessboard corners
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # find the rotation and translation vectors
        _, rvecs2, tvecs2, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # rvecs2 must be converted to the standard form 3x3
        R, J = cv2.Rodrigues(rvecs2)

        print(color.BOLD + color.GREEN + 'R Matrix: \n' + str(R) + color.END)
        print(color.BOLD + color.GREEN + 't Matrix: \n' + str(tvecs2) + color.END)
        return rvecs2, tvecs2, corners2, R

def initial_calibration(path):
    ''' Function to calculate the intrinsic parameters of camera
    according to a given number of calibration images stored on disk. '''

    # define the criteria used to calculate precisely the chessboard corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # according to chessboard dimensions (in our case we used a chessboard of
    # 10-1 squares along x and 7-1 squares along y)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # load calibration images from folder
    # this loads all the images contained in the folder!
    images = glob.glob(path + '*.png')

    for fname in images:
        # iterates all the images
        img = cv2.imread(fname)
        # our camera was mounted rotated of 180 deg along Z axis, so we rotate the images
        # to our point of view. It is not a required step and the result of the calibration
        # does not change, but try to keep the same image modifications along the process
        # e. g. if I flip the image here, I have to flip it everytime I calculate something on the image
        img = cv2.flip(img, 0)
        # converts image to black and white
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # if the chessboard is found, add object points and image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # refinement of chessboard points according to criteria
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # draw and display the corners on the corresponding image
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            # if you want to display the images one at a time during the loop,
            # uncomment these two lines. The loop proceeds only when you press any
            # key of the keyboard!!

            #cv2.imshow('img',img)
            #cv2.waitKey(0)

    # needed if you displayed the images before
    #cv2.destroyAllWindows()

    # finally perform the camera calibration using the object points and the image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print(color.BOLD + color.GREEN + 'Camera Matrix: \n' + str(mtx) + color.END)
    print(color.BOLD + color.GREEN + 'Distortion: \n' + str(dist) + color.END)

    # calculates the total error of the performed calibration
    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print(color.BOLD + color.YELLOW + "Total error of calibration: " + str(mean_error/len(objpoints)) + color.END)

    return mtx, dist

def undistort_image(img, K, D):
    ''' Function to perform the undistortion of the original image according
    to the intrinsic camera calibration performed before. '''

    # gets height and width of the frame
    h, w = img.shape[:2]
    # finds the optimal camera matrix according to distortion correction
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    print(color.BOLD + color.CYAN + 'Optimal Matrix: ' + str(newcameramtx) + color.END)

    # undistorts the image
    dst = cv2.undistort(img, K, D, None, newcameramtx)
    # crop the image to remove black patches around the corners due to undistortion
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('undistorted_image.png', dst)

    return dst

def main():
    ''' Program that performs the following steps:
    STEP 1: calibrate intrinsic parameters of camera
    STEP 2: snap a photo of the actual workspace and finds undistorted image
    STEP 3: finds the reference system in both original and undistorted images
    STEP 4: draws the reference systems on both images
    STEP 5: saves YAML file
    '''

    ######### STEP 1
    # perform camera calibration: finds the intrinsics of the camera
    # K and the distortion coefficients D. Gives as input the initial path of the
    # folder containing the calibration images
    K, D = initial_calibration('calib/')

    ######### STEP 2
    # starts the Kinect object to snap a photo of the actual workspace
    # only RGB is needed so the other values are set to False
    kinect = Kinect(True, False, False, False)
    kinect.acquire()
    # saves the frame as a copy, needed to correctly work with cv images
    img = kinect.color_new.copy()
    # stops the device since only a snapshot was needed
    kinect.stop()
    # saves the acquired frame on disk, useful to debug
    cv2.imwrite('workspace_image.png', img)

    dst = undistort_image(img, K, D)

    ######### STEP 3
    # we want to crop flip the original acquired frame to reduce background
    # noise. The actual size of the crop depends on where the user workspace is
    # in the original frame, in our case was a green pane. We manually found
    # suitable corners from the original frame and perform the cut for both the
    # undistorted image dst and the original image img
    dst = cv2.flip(dst, 0)
    dst = dst[0:900, 300:1800]

    img = cv2.flip(img, 0)
    img = img[0:900, 300:1800]

    # to check the cut, show the image
    cv2.imshow('Cropped Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ######### STEP 4
    # finds the reference point (0,0) of the calibration master positioned on the
    # workspace. This is the position of reference system H, so place it carefully
    print(color.BOLD + color.PURPLE + 'Find R and t of ORIGINAL image:' + color.END)
    rvecs1, tvecs1, corners1, R1 = findRT(img, K, D)

    # trova dst
    print(color.BOLD + color.PURPLE + 'Find R and t of UNDISTORTED image:' + color.END)
    rvecs2, tvecs2, corners2, R2 = findRT(dst, K, D)

    ######### STEP 5
    # project 3D points to image plane. This is to show the position of the
    # reference point calculated before!
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    # points on the original image
    imgpts, jac = cv2.projectPoints(axis, rvecs1, tvecs1, K, D)
    img = draw(img, corners1, imgpts)######### STEP 1
    cv2.imshow('Reference System on Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('original_reference.png', img)

    # points on the original image
    imgpts, jac = cv2.projectPoints(axis, rvecs2, tvecs2, K, D)
    dst = draw(dst, corners2, imgpts)
    cv2.imshow('Reference System on Undistorted Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('undistorted_reference.png', dst)

    ######### STEP 6
    # save the YAML file of the calibration!

    dictionary = [{'K': K.tolist()}, {'D' : D.tolist()}, {'R' : R1.tolist()}, {'t' : tvecs1.tolist()}, {'Rd' : R2.tolist()}, {'td' : tvecs2.tolist()}]
    dict2yaml(dictionary, 'camera_calibration.yaml')


if __name__ == '__main__':
    main()
