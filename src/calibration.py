#!/usr/bin/env python

import glob
import cv2
import numpy as np
import argparse
import utils
import graphical_utils as gu

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

def calibrate_pic(image, sq_x, sq_y, chess_size):
    # Funzione per calibrare un'immagine singola.

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (sq_x, sq_y), None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((sq_y*sq_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:sq_x, 0:sq_y].T.reshape(-1, 2) * chess_size

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return corners2, ret, objp
    else:
        return None, ret, None
    
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
        masked_img = img.copy()
        #masked_img = maskImg(img.copy())

        corners2, ret, objp = calibrate_pic(masked_img, sq_x, sq_y, chess_size)

        # if the chessboard is found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners2)

            # draw and display the corners on the corresponding image
            masked_img = cv2.drawChessboardCorners(masked_img, (sq_x, sq_y), corners2, ret)

            if debug:
                # show pattern on image only if debug is active
                cv2.imshow('Chessboard pattern', masked_img)
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

def calibrate_camera(x_size, y_size, chess_size, calib_folder, debug):
    ''' Programma che esegue i seguenti passaggi:
    STEP 1: calibra i parametri intrinseci della telecamera
    STEP 2: elabora le immagini di calibrazione e salva i risultati su un file YAML.

    INPUTS:
    - x_size: dimensione del pattern della scacchiera lungo la dimensione x
    - y_size: dimensione del pattern della scacchiera lungo la dimensione y
    - chess_size: dimensione di un singolo quadrato della scacchiera in mm
    - calib_folder: directory contenente le immagini di calibrazione
    - debug: flag per attivare la modalità di debug
    '''

    print('-- STARTING A CALIBRATION SESSION WITH PARAMETERS: --')
    print('SQUARES ALONG X:', x_size)
    print('SQUARES ALONG Y:', y_size)
    print('SQUARES SIZE:', chess_size)
    print('DEBUG DIRECTORY IS ./debug, YAML SAVED IN ./yaml')

    x_size -= 1
    y_size -= 1

    K, D = initial_calibration(calib_folder, x_size, y_size, chess_size, debug)

    dictionary = {'K': K.tolist(), 'D': D.tolist()}
    utils.dict2yaml(dictionary, './yaml/camera_calibration.yaml')

    print('-- CALIBRATION COMPLETE! --')
    print('-- CAMERA PARAMETERS --')
    print('K:', dictionary['K'])
    print('D:', dictionary['D'])

def args_preprocess():
    ''' Funzione che analizza gli argomenti passati dalla riga di comando e li imposta come variabili
    per la funzione principale. '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'x_size', type=int, help='Specifies number of squares in chessboard along x dimension. Count internal ones from black square to white square!')
    parser.add_argument(
        'y_size', type=int, help='Specifies number of squares in chessboard along y dimension. Count internal ones from black square to white square!')
    parser.add_argument(
        'chess_size', type=float, help='Size of a single square of chessboard in mm.')
    parser.add_argument(
        'calib_folder', type=str, help='Path of the calibration folder containing calibration images.')
    parser.add_argument(
        '--debug', action='store_true', help='Optional debug flag, set to true if present.')

    args = parser.parse_args()

    calibrate_camera(args.x_size, args.y_size, args.chess_size, args.calib_folder, args.debug)

if __name__ == '__main__':
    args_preprocess()
