import numpy as np
import cv2
import glob
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

class Kinect():
    ''' Kinect object, it uses pylibfreenect2 as interface to get the frames.
    The original example was taken from the pylibfreenect2 github repository, at:
    https://github.com/r9y9/pylibfreenect2/blob/master/examples/selective_streams.py  '''

    def __init__(self, enable_rgb, enable_depth, need_bigdepth, need_color_depth_map):
        ''' Init method called upon creation of Kinect object '''

        # according to the system, it loads the correct pipeline
        # and prints a log for the user
        try:
            from pylibfreenect2 import OpenGLPacketPipeline
            self.pipeline = OpenGLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenCLPacketPipeline
                self.pipeline = OpenCLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline
                self.pipeline = CpuPacketPipeline()

        #rospy.loginfo(color.BOLD + color.YELLOW + '-- PACKET PIPELINE: ' + str(type(self.pipeline).__name__) + ' --' + color.END)

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        # creates the freenect2 device
        self.fn = Freenect2()
        # if no kinects are plugged in the system, it quits
        self.num_devices = self.fn.enumerateDevices()
        if self.num_devices == 0:
            #rospy.loginfo(color.BOLD + color.RED + '-- ERROR: NO DEVICE CONNECTED!! --' + color.END)
            sys.exit(1)

        # otherwise it gets the first one available
        self.serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(self.serial, pipeline=self.pipeline)

        # defines the streams to be acquired according to what the user wants
        types = 0
        if self.enable_rgb:
            types |= FrameType.Color
        if self.enable_depth:
            types |= (FrameType.Ir | FrameType.Depth)
        self.listener = SyncMultiFrameListener(types)

        # Register listeners
        if self.enable_rgb:
            self.device.setColorFrameListener(self.listener)
        if self.enable_depth:
            self.device.setIrAndDepthFrameListener(self.listener)

        if self.enable_rgb and self.enable_depth:
            self.device.start()
        else:
            self.device.startStreams(rgb=self.enable_rgb, depth=self.enable_depth)

        # NOTE: must be called after device.start()
        if self.enable_depth:
            self.registration = Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

        # last number is bytes per pixel
        self.undistorted = Frame(512, 424, 4)
        self.registered = Frame(512, 424, 4)

        # Optinal parameters for registration
        self.need_bigdepth = need_bigdepth
        self.need_color_depth_map = need_color_depth_map

        if self.need_bigdepth:
            self.bigdepth = Frame(1920, 1082, 4)
        else:
            self.bigdepth = None

        if self.need_color_depth_map:
            self.color_depth_map = np.zeros((424, 512),  np.int32).ravel()
        else:
            self.color_depth_map = None


    def acquire(self):
        ''' Acquisition method to trigger the Kinect to acquire new frames. '''

        # acquires a frame only if it's new
        frames = self.listener.waitForNewFrame()

        if self.enable_rgb:
            self.color = frames["color"]
            self.color_new = cv2.resize(self.color.asarray(), (int(1920 / 1), int(1080 / 1)))
            # The image obtained has a fourth dimension which is the alpha value
            # thus we have to remove it and take only the first three
            self.color_new = self.color_new[:,:,0:3]
            # the kinect sensor mirrors the images, so we have to flip them back
            #self.color_new = cv2.flip(self.color_new, 0)
            #self.color_new = cv2.flip(self.color_new, 1)
            #self.color_new = self.color_new[230:830, 540:1450]
        if self.enable_depth:
            # these only have one dimension, we just need to convert them to arrays
            # if we want to perform detection on them
            self.depth = frames["depth"]
            #rospy.loginfo(self.depth.asarray() / 4500.)
            self.depth_new = cv2.resize(self.depth.asarray() / 4500., (int(512 / 1), int(424 / 1)))
            self.depth_new = cv2.flip(self.depth_new, 1)

            self.registration.undistortDepth(self.depth, self.undistorted)
            self.undistorted_new = cv2.resize(self.undistorted.asarray(np.float32) / 4500., (int(512 / 1), int(424 / 1)))
            self.undistorted_new = cv2.flip(self.undistorted_new, 0)
            self.undistorted_new = cv2.flip(self.undistorted_new, 1)

            self.ir = frames["ir"]
            self.ir_new = cv2.resize(self.ir.asarray() / 65535., (int(512 / 1), int(424 / 1)))
            self.ir_new = cv2.flip(self.ir_new, 0)
            self.ir_new = cv2.flip(self.ir_new, 1)

        if self.enable_rgb and self.enable_depth:
            self.registration.apply(self.color, self.depth, self.undistorted, self.registered,
                                    bigdepth=self.bigdepth, color_depth_map=self.color_depth_map)
            # RGB + D
            self.registered_new = self.registered.asarray(np.uint8)
            self.registered_new = cv2.flip(self.registered_new, 0)
            self.registered_new = cv2.flip(self.registered_new, 1)

            if self.need_bigdepth:
                self.bigdepth_new = cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1)))
                self.bigdepth_new = cv2.flip(self.bigdepth_new, 0)
                self.bigdepth_new = cv2.flip(self.bigdepth_new, 1)
                #rospy.loginfo(self.bigdepth_new[0])
                #rospy.loginfo(self.color_new[0])
                #cv2.imshow("bigdepth", cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1))))
            if self.need_color_depth_map:
                #cv2.imshow("color_depth_map", self.color_depth_map.reshape(424, 512))
                self.color_depth_map_new = self.color_depth_map.reshape(424, 512)
                self.color_depth_map_new = cv2.flip(self.color_depth_map, 0)
                self.color_depth_map_new = cv2.flip(self.color_depth_map, 1)

        # do this anyway to release every acquired frame
        self.listener.release(frames)

    def stop(self):
        #rospy.loginfo(color.BOLD + color.RED + '\n -- CLOSING DEVICE... --' + color.END)
        self.device.stop()
        self.device.close()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calib/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('Camera Matrix: ' + str(mtx))
print('Distortion: ' + str(dist))

# undistort image

kinect = Kinect(True, False, False, False)
kinect.acquire()
img = kinect.color_new
kinect.stop()

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# find reference point

#img = cv2.imread('big3.png')

img = cv2.flip(dst, 0)
img = img[230:830, 540:1450]

# find R and T in the reference
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

if ret == True:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.
    _, rvecs2, tvecs2, inliers = cv2.solvePnPRansac(objp, corners2, newcameramtx, dist)
    R, J = cv2.Rodrigues(rvecs2)
    print('R Matrix: ' + str(R))
    print('T Matrix: ' + str(tvecs2))

    # project 3D points to image plane
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs2, tvecs2, newcameramtx, dist)

    img = draw(img,corners2,imgpts)
    cv2.imshow('img',img)
    k = cv2.waitKey(0) & 0xff
    if k == 's':
        cv2.imwrite('ref.png', img)

cv2.destroyAllWindows()


mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], newcameramtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print "total error: ", mean_error/len(objpoints)
