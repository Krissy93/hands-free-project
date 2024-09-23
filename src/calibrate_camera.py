#!/usr/bin/env python

import glob
import cv2
import numpy as np
# custom library
import graphical_utils as gu
import argparse
import utils

''' WARNING: change camera to Camera if a normal camera is used,
otherwise set it to Kinect if a Kinect v2 device is used.

x_size and y_size set the chessboard sizes. The actual squares used for the grid are the
internal ones counting from black square to white square! Chess size is in millimeters.
Workspace contains top-left xy point and bottom-right xy point of workspace to be drawn.
BE SURE TO USE THE CORRECT VALUES ACCORDING TO THE CHESSBOARD USED DURING ACQUISITION

HOW TO LAUNCH:
python3 calibrate_camera.py \
camera Kinect \
x_size 10 \
y_size 7 \
chess_size 23.1 \
calib_folder './calib_img_old' \
mode 'acquisition' \
--workspace '266,139,1074,663' \
--debug
'''

def calibrate_pic(image, sq_x, sq_y, chess_size):
    ''' Function to calibrate a single image. It founds the chessboard corners
        and performs a corners refinement if successful.

    INPUTS
    - image: image to be calibrated
    - sq_x: squares of the chessboard along x dimension
    - sq_y: squares of the chessboard along y dimension
    - chess_size: size in mm
    - pt1: top_left point of square used to draw the rectangle around the workspace
    - pt2: bottom_right point of square used to draw the rectangle around the workspace

    OUTPUTS:
    - corners2: chessboard corners refined after calibration procedure
    - ret: success flag of procedure, true if successful
    '''

    # converts image to black and white
    # Converti l'immagine in scala di grigi in un'immagine a colori
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    print(f'Chessboard: ({sq_x}, {sq_y})')

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (sq_x, sq_y), None)

    # if the chessboard is found, add object points and image points (after refining them)
    if ret:
        # define the criteria used to calculate precisely the chessboard corners
        # first term defines the termination criteria type, in this case it stops
        # if it reaches the desired epsilon or the max number of iterations.
        # max iterations in this example are 30 (second term)
        # epsilon in this example is 0.001 (third term)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # according to chessboard dimensions
        objp = np.zeros((sq_y*sq_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:sq_x, 0:sq_y].T.reshape(-1, 2) * chess_size

        # find out the precise positions of chessboard corners
        # the fourth item is the zero zone equal to half of the size of the dead region
        # in the middle of the search zone over which the summation in the formula
        # is not done. It is used sometimes to avoid possible singularities
        # of the autocorrelation matrix. The value of (-1,-1) indicates that
        # there is no such a size.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return corners2, ret, objp

    else:
        return None, ret, None

def show_calibration(image, sq_x, sq_y, chess_size, pt1, pt2):
    ''' Function used to visualize in real-time if the image shown on video is okay
    for calibration purposes. It draws on it the chessboard pattern found after calibration
    and the workspace size defined by pt1 and pt2. This image is not saved!

    INPUTS
    - image: image to be calibrated
    - sq_x: squares of the chessboard along x dimension
    - sq_y: squares of the chessboard along y dimension
    - chess_size: size in mm
    - pt1: top_left point of square used to draw the rectangle around the workspace
    - pt2: bottom_right point of square used to draw the rectangle around the workspace

    OUTPUTS:
    - image: image on which the workspace and the chessboard corners have been drawn
             (if the procedure is successful)
    '''

    image = image.copy()
    corners2, ret, _ = calibrate_pic(image, sq_x, sq_y, chess_size)

    if ret:
        # draw and display the corners on the corresponding image
        image = cv2.drawChessboardCorners(image, (sq_x, sq_y), corners2, ret)

    # draw white rectangle to correctly center the camera
    # it gives the top-left point and the bottom-right point of the rectangle,
    # the color in BGR and the line thickness
    cv2.rectangle(image, pt1, pt2, (255, 255, 255), 1)

    return image

def findRT(img, mtx, dist, sq_x, sq_y, chess_size):
    ''' Function to extract R and t corresponding to the fixed reference system
    placed on the user workspace. This is the reference system origin used to
    calculate and to which refer the real marker positions (e. g. the index finger positions)
    with respect to a known origin for the user, which is different than the image reference.

    INPUTS:
    - img: image on which the chessboard should be found
    - mtx: camera matrix found after calibration
    - dist: distortion coefficients found after calibration
    - sq_x: squares of the chessboard along x dimension
    - sq_y: squares of the chessboard along y dimension
    - chess_size: size in mm

    OUTPUTS:
    - rvecs2: rotation vector found after ransac computation in opencv format
    - tvecs2: translation vector 1x3 found after ransac computation
    - corners2: corners of the chessboard area used to compute R and T
    - R: rotation vector equal to rvecs2 but in classic 3x3 format
    '''

    img = img.copy()
    corners2, ret, objp = calibrate_pic(img, sq_x, sq_y, chess_size)

    # if the chessboard has been found, calculates R and t
    if ret:
        # find the rotation and translation vectors
        _, rvecs2, tvecs2, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # converts rvecs2 to the standard form 3x3
        R, J = cv2.Rodrigues(rvecs2)

        return rvecs2, tvecs2, corners2, R
    else:
        return None, None, None, None

def maskImg(img):
    # qui la maschero
    '''1) creo con mask = np.zeros() una matrice di dimensioni uguali a quelle della mia img
        2) ho preso i due punti del workspace top-left (1), bottom-right (2)
        3) tratto la maschera come una matrice di binari, quindi mask[x1:x2, y1:y2] = 1
        4) frame = img.copy()
        5) masked_image1 = mask * frame[:,:,0], 
        6) riconcatenare l'immagine come masked_image1,2,3 (guarda su google)
    '''

    # 1) Creo una maschera di dimensioni uguali all'immagine
    mask = np.zeros_like(img[:,:,0])

    #2) Definisco i punti del workspace: top-left (x1, y1) e bottom-right (x2, y2)
    x1, y1 = 92, 657
    x2, y2 = 953, 56
    
    # Inverti l'asse y
    HEIGHT = img.shape[0]
    y1 = HEIGHT - y1
    y2 = HEIGHT - y2

    # 3) Imposto la maschera come una matrice binaria
    mask[y1:y2, x1:x2] = 1

    # 4) Copio l'immagine originale per mantenere l'originale inalterata
    frame = img.copy()

    # 5) Applico la maschera all'immagine per ottenere una versione mascherata dell'immagine
    masked_image1 = cv2.bitwise_and(frame[:,:,0], frame[:,:,0], mask=mask)  # Canale R
    masked_image2 = cv2.bitwise_and(frame[:,:,1], frame[:,:,1], mask=mask)  # Canale G
    masked_image3 = cv2.bitwise_and(frame[:,:,2], frame[:,:,2], mask=mask)  # Canale B

    # 6) Riconcateno le immagini mascherate
    # Concatenazione lungo l'asse delle colonne (orizzontale)
    masked_image = np.dstack((masked_image1, masked_image2, masked_image3))
    
    return masked_image


def initial_calibration(path, sq_x, sq_y, chess_size, debug):
    ''' Function to calculate the intrinsic parameters of the camera
    according to a given number of calibration images stored on disk.

    INPUTS:
    - path: full path to folder containing all images needed to calibrate camera
    - sq_x: squares of the chessboard along x dimension
    - sq_y: squares of the chessboard along y dimension
    - chess_size: size in mm
    - debug: flag needed to eventually show result of chessboard on images

    OUTPUTS:
    - mtx: camera matrix found after calibration
    - dist: camera distortion coefficients found after calibration
    '''

    print(gu.Color.BOLD + gu.Color.CYAN + '-- STARTING INITIAL CALIBRATION. LOADING IMAGES... --' + gu.Color.END)

    # load all calibration images from folder
    images = glob.glob(path + '/*.png')

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    print(f'Ho caricato {len(images)} immagini')
    print(f'Chessboard: ({sq_x}, {sq_y})')

    for fname in images:

        img = cv2.imread(fname)

        masked_img = maskImg(img)

        corners2, ret, objp = calibrate_pic(img, sq_x, sq_y, chess_size)

        # if the chessboard is found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners2)

            # draw and display the corners on the corresponding image
            img = cv2.drawChessboardCorners(img, (sq_x, sq_y), corners2, ret)

            if debug:
                # show pattern on image only if debug is active
                cv2.imshow('Chessboard pattern', img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    # perform the camera calibration using the object points and the image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (masked_img.shape[1], masked_img.shape[0]), None, None)

    # calculates the total error of the performed calibration
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print(gu.Color.BOLD + gu.Color.YELLOW + "MEAN REPROJECTION ERROR OF CALIBRATION: " + str(mean_error/len(objpoints)) + gu.Color.END)

    return mtx, dist


def calibrate_camera(camera, x_size, y_size, chess_size, calib_folder, debug):
    ''' Program that performs the following steps:
    STEP 1: calibrate intrinsic parameters of camera
    STEP 2: snap a photo of the actual workspace and finds undistorted image
    STEP 3 -- Optional: crop and show both images
    STEP 4: finds the reference system in both original and undistorted images
    STEP 5: draws the reference systems on both images
    STEP 6: saves YAML file of camera intrinsic and distortion parameters, R and t
            of reference system centered on the user frame needed to convert from px to m

    INPUTS:
    - camera: camera object according to user defined flag (may be Camera or Kinect)
    - x_size: squares of the chessboard along x dimension
    - y_size: squares of the chessboard along y dimension
    - chess_size: checkerboard single square size in mm
    - calib_folder: directory containing the images needed to calibrate camera
    - debug: flag needed to eventually show result of chessboard on images

    OUTPUTS:
    - YAML file containing the camera intrinsic matrix, the distortion coefficients,
      the R matrix and t vector of reference system needed to convert from px (camera frame)
      to m (user frame centered on point 0 of chessboard)
    '''

    print(gu.Color.BOLD + gu.Color.PURPLE + '-- STARTING A CALIBRATION SESSION WITH PARAMETERS: --' + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'SQUARES ALONG X: ' + str(x_size) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'SQUARES ALONG Y: ' + str(y_size) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'SQUARES SIZE: ' + str(chess_size) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.CYAN + 'DEBUG DIRECTORY IS ./debug, YAML SAVED IN ./yaml' + gu.Color.END)

    # for example, if we use a chessboard of 10 squares along x and 7 squares along y
    # the actual squares used for the grid are the internal ones counting from
    # black square to white square, hence we remove 1 from both (and not 2)
    x_size -= 1
    y_size -= 1

    ######### STEP 1: CALIBRATE INTRINSIC PARAMETERS OF CAMERA
    K, D = initial_calibration(calib_folder, x_size, y_size, chess_size, debug)

    ######### STEP 2: SNAP A PHOTO OF THE ACTUAL WORKSPACE AND FIND UNDISTORTED IMAGE
    camera.acquire(False)
    # saves the frame as a copy, needed to correctly work with cv images
    img = camera.RGBundistorted.copy()
    # stops the device since only a snapshot is needed
    camera.stop()
    # saves the acquired frame on disk, useful to debug
    cv2.imwrite('./debug/workspace_image.png', img)

    ######### STEP 3: CROP AND SHOW ORIGINAL AND UNDISTORTED IMAGES
    # we want to crop the original acquired frame to reduce background
    # noise. The actual size of the crop depends on where the user workspace is
    # in the original frame. We manually found suitable corners from the original frame and perform
    # the cut for both the undistorted image dst and the original image img
    #print(utils.Color.BOLD + utils.Color.PURPLE + '-- CROPPING THE IMAGE TO REDUCE NOISE --' + utils.Color.END)
    #pt1 = (266, 139)
    #pt2 = (1074, 663)
    #print(utils.Color.BOLD + utils.Color.PURPLE + 'STARTING POINT OF CROP COORDINATES (TOP-LEFT): ' + str(pt1) + utils.Color.END)
    #print(utils.Color.BOLD + utils.Color.PURPLE + 'ENDING POINT OF CROP COORDINATES (BOTTOM-RIGHT): ' + str(pt2) + utils.Color.END)
    #img = utils.crop(img, pt1, pt2)

    # if debug:
    #     # to check the cut, show the image img
    #     cv2.imshow('Cropped Image', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    ######### STEP 4: FINDS REFERENCE SYSTEM IN BOTH ORIGINAL AND UNDISTORTED IMAGE
    # finds the reference point (0,0) of the calibration master positioned on the
    # workspace. This is the position of reference system H, so place it carefully
    print(gu.Color.BOLD + gu.Color.PURPLE + '-- FINDING REFERENCE SYSTEM... --' + gu.Color.END)

    rvecs1, tvecs1, corners1, R1 = findRT(img, K, D, x_size, y_size, chess_size)

    ######### STEP 5: SHOW REFERENCE SYSTEM ON BOTH IMAGES
    # project 3D points to image plane. This is to show the position of the
    # reference point calculated before!
    axis = np.float32([[20, 0, 0], [0, 20, 0], [0, 0, -20]]).reshape(-1, 3)
    # points on the original image
    imgpts, jac = cv2.projectPoints(axis, rvecs1, tvecs1, K, D)
    gu.draw_reference(img, corners1, imgpts)

    cv2.imshow('Reference System on Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./debug/original_reference.png', img)

    ######### STEP 6: SAVES OUTPUT
    # saves output on yaml file and prints on screen the calibration matrix
    dictionary = {'K': K.tolist(), 'D': D.tolist(), 'R': R1.tolist(), 't': tvecs1.tolist()}
    utils.dict2yaml(dictionary, './yaml/camera_calibration.yaml')

    print(gu.Color.BOLD + gu.Color.GREEN + '-- CALIBRATION COMPLETE! --' + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + '-- CAMERA PARAMETERS --' + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + 'K: ' + str(dictionary['K']) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + 'D: ' + str(dictionary['D']) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + '-- ORIGINAL IMAGE CALIBRATION MATRIX --' + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + 'R: ' + str(dictionary['R']) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.GREEN + 't: ' + str(dictionary['t']) + gu.Color.END)

def frames_acquisition(camera, x_size, y_size, chess_size, workspace, saving_dir):
    ''' Program used to take photo of the workspace to use it for calibration. It shows in real-time the corners found
    on the master in order to know if the setup is good. Pressing 's' it automatically takes a photo and store it in the
    ./calib_img directory.

    INPUTS:
    - camera: camera object according to user defined flag (may be Camera or Kinect)
    - x_size: squares of the chessboard along x dimension
    - y_size: squares of the chessboard along y dimension
    - chess_size: checkerboard single square size in mm
    - workspace: list of xy points of top-left (tl) and bottom-right (br) corners of user workspace
                 defined as [xtl, ytl, xbr, ybr] and needed to check if pics are taken correctly
    - saving_dir: directory containing the images saved by this program and used to calibrate camera

    OUTPUTS:
    - saves images in the saving directory that will be used to calibrate the camera
    '''

    print(gu.Color.BOLD + gu.Color.PURPLE + '-- ACQUIRING CALIBRATION IMAGES WITH PARAMETERS: --' + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'SQUARES ALONG X: ' + str(x_size) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'SQUARES ALONG Y: ' + str(y_size) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'SQUARES SIZE: ' + str(chess_size) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'TOP-LEFT POINT OF WORKSPACE: ' + str((workspace[0], workspace[1])) + gu.Color.END)
    print(gu.Color.BOLD + gu.Color.PURPLE + 'BOTTOM-RIGHT POINT OF WORKSPACE: ' + str((workspace[2], workspace[3])) + gu.Color.END)

    print(gu.Color.CYAN + '-- PRESS ENTER WHEN FRAME IS OK, THEN PRESS S TO SAVE IMAGE, Q TO QUIT --' + gu.Color.END)

    # image counter
    i = 0

    while(1):

        while(1):
            # capture a frame
            camera.acquire(False)
            frame = camera.RGBundistorted.copy()

            # creates a copy of the frame
            frame_copy = frame.copy()
            #frame_copy= cv2.flip(frame_copy, 1)

            # shows chessboard pattern on image to help users set the chessboard
            # correctly inside the workspace area during the calibration procedure.
            # this image is NOT saved, is only visualized for help purposes!
            show_calibration(frame_copy, x_size, y_size, chess_size, (workspace[0], workspace[1]), (workspace[2], workspace[3]))
            # show pattern on image
            cv2.imshow('Image acquisition', frame_copy)
            # waits until the user doesn't press enter to proceed
            if cv2.waitKey(10) & 0xFF == 13:
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # creates a copy of the frame

        #frame_copy = frame.copy()
        #frame_copy= cv2.flip(frame_copy, 1)
        #cv2.imshow('Image acquisition', frame_copy)

        # show original image
        cv2.imshow('Image acquisition', frame)
        
        # waits for the user to press a key, that must be exit or s
        k = cv2.waitKey(0)
        if k == 113:
            # wait for ESC key to exit
            print(gu.Color.BOLD + gu.Color.RED + '\n-- EXIT --\n' + gu.Color.END)
            cv2.destroyAllWindows()
            break
        elif k == 115:
            # wait for 's' key to save an image
            # it saves the original frame, not the modified one with pattern on it!
            cv2.imwrite(saving_dir + '/master' + str(i) + '.png', frame)

            #cv2.imwrite(saving_dir + '/master' + str(i) + '.png', frame_copy)

            print(gu.Color.GREEN + '-- IMAGE ' + str(i) + ' SAVED --' + gu.Color.END)
            cv2.destroyAllWindows()
            # updates image number
            i = i + 1
        else:
            print(gu.Color.BOLD + gu.Color.RED + '\n-- INVALID COMMAND --\n' + gu.Color.END)
            cv2.destroyAllWindows()
    
    camera.stop()

def args_preprocess():
    ''' Function that parses the arguments passed by command line and sets them as variables
    for the main function. '''

    # build arguments parser
    parser = argparse.ArgumentParser()
    # adds arguments accordingly
    parser.add_argument(
        'camera', choices = ['Camera', 'Kinect'], help='Specifies the type of camera used. Possible values: Kinect or Camera.')
    parser.add_argument(
        'x_size', type=int, help='Specifies number of squares in chessboard along x dimension. Count internal ones from black square to white square!')
    parser.add_argument(
        'y_size', type=int, help='Specifies number of squares in chessboard along y dimension. Count internal ones from black square to white square!')
    parser.add_argument(
        'chess_size', type=float, help='Size of a single square of chessboard in mm.')
    parser.add_argument(
        'calib_folder', type=str, help='Path of the calibration folder containing calibration images.')
    parser.add_argument(
        'mode', choices = ['acquisition', 'calibration', 'full'], help='Defines modality of execution.')
    parser.add_argument(
        '--workspace', type=str, default=[], help='List of workspace points, top-left xy and bottom-right xy, delimited by a comma.')
    parser.add_argument(
        '--debug', action='store_true', help='Optional debug flag, set to true if present.')

    args = parser.parse_args()

    if args.camera == 'Kinect':
        camera = utils.Kinect(enable_rgb=True, enable_depth=False, need_bigdepth=False, need_color_depth_map=False)
    else:
        camera = utils.Camera(enable_rgb=True)

    if len(args.workspace) > 0:
        workspace = [int(item) for item in args.workspace.split(',')]

    if args.mode == 'acquisition':
        frames_acquisition(camera, args.x_size, args.y_size, args.chess_size, workspace, args.calib_folder)
    elif args.mode == 'calibration':
        calibrate_camera(camera, args.x_size, args.y_size, args.chess_size, args.calib_folder, args.debug)
    else:
        frames_acquisition(camera, args.x_size, args.y_size, args.chess_size, workspace, args.calib_folder)
        calibrate_camera(camera, args.x_size, args.y_size, args.chess_size, args.calib_folder, args.debug)

if __name__ == '__main__':
    args_preprocess()
