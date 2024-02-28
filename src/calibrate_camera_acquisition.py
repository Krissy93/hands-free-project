#!/usr/bin/env python

import copy
import cv2
import utils
import numpy as np


def main():
    ''' Program used to take photo of the workspaced to use it for calibration. It shows in real-time the corners found
    on the master in order to know if setup is good. Pressing 's' it automatically takes a photo and store it in the
    ./calib_img directory. '''

    camera = utils.Camera(True)

    i = 0
    while True:
        # Capture frame
        camera.acquire()
        frame = camera.color_new.copy()

        ### CHESSBOARD AND RECTANGLE
        frame_copy = copy.deepcopy(frame)

        # define the criteria used to calculate precisely the chessboard corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # according to chessboard dimensions
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # converts image to black and white
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # if the chessboard is found, add object points and image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # refinement of chessboard points according to criteria
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # draw and display the corners on the corresponding image
            frame_copy = cv2.drawChessboardCorners(frame_copy, (9, 6), corners2, ret)

        # Draw white rectangle for center the camera
        cv2.rectangle(frame_copy, (utils.x1, utils.y1), (utils.x2, utils.y2), (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Image acquisition', frame_copy)

        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            print(utils.Color.BOLD + utils.Color.YELLOW + '\n-- EXIT --\n' + utils.Color.END)
            break
        elif k == ord('s'):  # wait for 's' key to save an image
            cv2.imwrite('./calib_img/master' + str(i) + '.png', frame)
            print(utils.Color.CYAN + '-- IMAGE master' + str(i) + ' ACQUIRED --' + utils.Color.END)
            i = i + 1

    camera.stop()


if __name__ == '__main__':
    main()
