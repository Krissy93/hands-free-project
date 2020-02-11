#!/usr/bin/env python
from __future__ import division
import os
# ONLY PRINTS WARNINGS AND ERRORS
os.environ['GLOG_minloglevel'] = '2'
import rospy
import time
import math

import numpy as np
from scipy.spatial import distance
import cv2
import sys
import caffe
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from geometry_msgs.msg import Pose
from cartesian import *
# sawyer interface
import intera_interface

#######

# defines keypoint pairs
global POSE_PAIRS, K, R, D, t, reference
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

K = np.array([[1.06314155e+03, 0.00000000e+00, 9.62163918e+02],
              [0.00000000e+00, 1.06450171e+03, 5.40462082e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D = np.array([[0.04147462, -0.03934306, -0.00013871, -0.00071449, -0.00889901]])
# Rt ottenute con calibrazione su immagine NON antidistorta
R = np.array([[0.99980291, -0.01924482, -0.00487729],
              [0.01907285, 0.99926855, -0.03314486],
              [0.00551159, 0.0330453, 0.99943866]])
t = np.array([[-32.91148716], [-15.32579363], [41.38506969]])
# Rt ottenute con calibrazione su immagine antidistorta
Rd = np.array([[0.99978689, -0.01612781, -0.01288714],
               [0.01586225, 0.99966504, -0.02044997],
               [0.01321263, 0.02024119, 0.99970782]])
td = np.array([[-34.13972018], [-15.92266493], [ 42.32613221]])

reference, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), Rd, td, K, D)
reference = reference.flatten()
reference = np.array([[reference[0], reference[1], 1.0]]) #px coordinates

################

class color:
   PURPLE = '\033[95m' #loginfo and commands
   CYAN = '\033[96m' #quando chiedo input
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m' #ok
   YELLOW = '\033[93m' #info
   RED = '\033[91m' #errori
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class Kinect():
    ''' Kinect object, it uses pylibfreenect2 as interface to get the frames.
    The original example was taken from the pylibfreenect2 github repository, at:
    https://github.com/r9y9/pylibfreenect2/blob/master/examples/selective_streams.py  '''

    def __init__(self, enable_rgb, enable_depth, need_bigdepth, need_color_depth_map, K, D):
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

        rospy.loginfo(color.BOLD + color.YELLOW + '-- PACKET PIPELINE: ' + str(type(self.pipeline).__name__) + ' --' + color.END)

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.K = K
        self.D = D

        # creates the freenect2 device
        self.fn = Freenect2()
        # if no kinects are plugged in the system, it quits
        self.num_devices = self.fn.enumerateDevices()
        if self.num_devices == 0:
            rospy.loginfo(color.BOLD + color.RED + '-- ERROR: NO DEVICE CONNECTED!! --' + color.END)
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
            #self.color_new = self.color_new[0:1080, 520:1650]
            self.correct_distortion()
        if self.enable_depth:
            # these only have one dimension, we just need to convert them to arrays
            # if we want to perform detection on them
            self.depth = frames["depth"]
            #rospy.loginfo(self.depth.asarray() / 4500.)
            self.depth_new = cv2.resize(self.depth.asarray() / 4500., (int(512 / 1), int(424 / 1)))
            #self.depth_new = cv2.flip(self.depth_new, 1)

            self.registration.undistortDepth(self.depth, self.undistorted)
            self.undistorted_new = cv2.resize(self.undistorted.asarray(np.float32) / 4500., (int(512 / 1), int(424 / 1)))
            #self.undistorted_new = cv2.flip(self.undistorted_new, 0)
            #self.undistorted_new = cv2.flip(self.undistorted_new, 1)

            self.ir = frames["ir"]
            self.ir_new = cv2.resize(self.ir.asarray() / 65535., (int(512 / 1), int(424 / 1)))
            #self.ir_new = cv2.flip(self.ir_new, 0)
            #self.ir_new = cv2.flip(self.ir_new, 1)

        if self.enable_rgb and self.enable_depth:
            self.registration.apply(self.color, self.depth, self.undistorted, self.registered,
                                    bigdepth=self.bigdepth, color_depth_map=self.color_depth_map)
            # RGB + D
            self.registered_new = self.registered.asarray(np.uint8)
            #self.registered_new = cv2.flip(self.registered_new, 0)
            #self.registered_new = cv2.flip(self.registered_new, 1)

            if self.need_bigdepth:
                self.bigdepth_new = cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1)))
                #self.bigdepth_new = cv2.flip(self.bigdepth_new, 0)
                #self.bigdepth_new = cv2.flip(self.bigdepth_new, 1)
                #rospy.loginfo(self.bigdepth_new[0])
                #rospy.loginfo(self.color_new[0])
                #cv2.imshow("bigdepth", cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1))))
            if self.need_color_depth_map:
                #cv2.imshow("color_depth_map", self.color_depth_map.reshape(424, 512))
                self.color_depth_map_new = self.color_depth_map.reshape(424, 512)
                #self.color_depth_map_new = cv2.flip(self.color_depth_map, 0)
                #self.color_depth_map_new = cv2.flip(self.color_depth_map, 1)

        # do this anyway to release every acquired frame
        self.listener.release(frames)

    def stop(self):
        rospy.loginfo(color.BOLD + color.RED + '\n -- CLOSING DEVICE... --' + color.END)
        self.device.stop()
        self.device.close()

    def correct_distortion(self):
        h,  w = self.color_new.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K,self.D,(w,h),1,(w,h))
        # undistort
        self.RGBundistorted = cv2.undistort(self.color_new, self.K, self.D, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        self.RGBundistorted = self.RGBundistorted[y:y+h, x:x+w]
        self.RGBundistorted = cv2.flip(self.RGBundistorted, 0)
        self.RGBundistorted = self.RGBundistorted[0:900, 520:1650]
#######

def init_workspace(kinect):
    ''' takes a snapshot of the workspace, finds the green rectangle and cuts the workspace
    according to the green dimension and the reference system '''

    kinect.acquire()
    frame = kinect.color_new
    # ## convert to hsv
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # detect green mask
    # #mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    # mask = cv2.inRange(hsv, (45, 100, 100), (70, 255,255))
    #
    # ## slice the green
    # imask = mask>0
    # green = np.zeros_like(frame, np.uint8)
    # green[imask] = frame[imask]
    # imgray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    # thresh, imgray = cv2.threshold(imgray, 127, 255, 0)
    #
    # _, contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # y, x
    #cut = frame[50:900, 200:1400]

    # saves
    cv2.imwrite("big3.png", frame)
    #cv2.imwrite("big2.png", frame)
    return frame

# ok
def init_network():
    ''' Function to initialize network parameters.
    Be sure to place the network and the weights in the correct folder. '''

    rospy.loginfo(color.BOLD + color.YELLOW + 'INITIALIZING CAFFE NETWORK...' + color.END)
    protoFile = "HandPose/hand/pose_deploy.prototxt"
    weightsFile = "HandPose/hand/pose_iter_102000.caffemodel"

    net = caffe.Net(protoFile, 1, weights=weightsFile)
    caffe.set_mode_gpu()

    # vecchio modo con cv2 troppo lento perche' non usa cuda
    #net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    #net.setPreferableBackend(DNN_BACKEND_CUDA)
    #net.setPreferableTarget(DNN_TARGET_CUDA)
    #setPreferableTarget(DNN_TARGET_CUDA_FP16)

    return net

#nok
def px2meters(pt, K, R, t):
    ''' Funzione di conversione tra pixel e metri per ottenere la posizione
    corretta in metri della posizione rilevata nell'immagine RGB-D. La Kinect
    prende (x,y) dall'RGB e (z) dal sensore di profondita', quindi deve essere
    calibrato per forza. Calibrazione fatta in MATLAB! Riporto qui le matrici '''

    K2 = np.linalg.inv(K)
    R2 = np.linalg.inv(R)
    pt = pt.T

    S = K2.dot(pt)
    #print('S = K-1 * pt: \n' + str(S))
    N = S - t
    #print('N = S - t: \n' + str(N))
    PT = R2.dot(N)
    #print('PT = R-1 * N: \n' + str(PT))
    #print(reference)
    XYZ = PT # in cm rispetto allo 0 videocamera

    return XYZ

def H2R(Ph):
    ''' conversione e movimentazione del robot
    gli viene passato il punto gia' convertito da px a m '''

    Ph = Ph.flatten()
    Ph = np.array([[Ph[1], Ph[0]]])
    print('Point is: ' + str(Ph))
    RW = np.array([[-0.0001, 0.0100], [-0.0100, -0.0001]])
    tW = np.array([[-0.0457],[0.3835]])
    Pr = RW.dot(Ph.T)
    Pr = Pr + tW
    Pr = Pr.flatten()
    print(Pr)

    #pr ha prima la z poi la y, quindi per muoverlo devo lanciare

    #limb = intera_interface.Limb('right')
    #limb.set_joint_position_speed(0.3)

    pose = Pose()
    pose.position.x = 0.7
    pose.position.y = round(Pr[1],2)
    pose.position.z = round(Pr[0],2)
    print('Pose: ' + str(pose))


    #joints = limb.ik_request(pose, end_point='right_hand', joint_seed=None,
    #               nullspace_goal=None, nullspace_gain=0.4,
    #               allow_collision=True)
    #print(joints)
    #limb.move_to_joint_positions(joints)

    move2cartesian(position=(pose.position.x, pose.position.y, pose.position.z), in_tip_frame=True, linear_speed=0.3)

    return Pr

def error(point, frame):
    '''funzione per trovare l'errore tra il punto dell'indice estratto
    e il punto calcolato da openpose. mi salva la coordinata dell'indice
    e fa una foto al ws '''

    files = next(os.walk('/home/optolab/ur2020_workspace/src/telemove/src/saved/'))[2]
    n = len(files)
    cv2.imwrite('/home/optolab/ur2020_workspace/src/telemove/src/saved/img_'+str(n+1)+'.png', frame)

    with open('/home/optolab/ur2020_workspace/src/telemove/src/points.txt', 'a') as file_object:
    # Append 'hello' at the end of file
        file_object.write('img_'+str(n+1)+'\t'+str(point)+'\n')

# ok
def hand_keypoints(net, frame, threshold, nPoints):
    ''' Chiama la rete per tirare fuori le coordinate del joint dell'indice,
    sia dx sia sx a seconda della mano utilizzata. Lo printa sull'immagine RGB
    mostrata a video come pallino a cui e' collegata la terna relativa (x,y,z).
    Quando e' presente anche il pollice, avvia l'acquisizione del punto reale dove
    deve muoversi il robot. Per ovviare al tremolio, prende 10 pti index e fa una media '''

    before = time.time()
    aspect_ratio = frame.shape[1]/frame.shape[0]
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # height frameshape0 width frameshape1
    net.blobs['image'].reshape(1, 3, inHeight, inWidth)
    net.blobs['image'].data[...] = inpBlob
    output = net.forward()
    output = output['net_output']
    # gets inference time required by network to perform the detection
    inference_time = time.time() - before

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

        # find global maxima of the probMap
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            # yellow circle on image
            #cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # writes red number of keypoint on image
            #cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    return points, inference_time

# cambia colori ma ok
def draw_skeleton(frame, points, draw, inference_time, G, H):
    ''' Function to draw the skeleton of one hand according to
    a pre-defined pose pair scheme to the frame. Does not return anything. '''

    A = frame.copy()

    # draw skeleton on frame
    if draw:
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            # if there is a point in both keypoints of the pair, draws the point and the connected line
            if points[partA] and points[partB]:
                cv2.line(A, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(A, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(A, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # the structure is: image, string of text, position from the top-left angle, font, size, BGR value of txt color, thickness, graphic
    cv2.putText(A, 'INFERENCE TIME: ' + str(inference_time) + ' SEC', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 65, 242), 3, cv2.LINE_AA)
    cv2.putText(A, str(G) + ' || HANDMAP: ' + str(H), (20,A.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 65, 242), 3, cv2.LINE_AA) #(20,980)
    # finally we display the image on a new window
    cv2.imshow('hand detection', A)

# ok
def closed_finger(points):
    ''' Function to check if the finger is closed or not, according to the position
    of the detected keypoints. If their position is relatively close (euclidean distance),
    then the finger is closed, otherwise is open. Returns a map of finger closed, where
    in position 0 there is the thumb and in position 4 the pinkie. 0 means closed,
    1 means opened finger. '''

    # thumb: coppia 0-4
    # index: coppia 5-8
    # middle: coppia 9-12
    # annular: coppia 13-16
    # pinkie: 17-20
    # sono sempre 4 coppie

    # scorre tutte le coppie
    handmap = []
    #print('points: ' + str(points))

    # check per definire il valore dei riferimenti se qualche keypoint manca
    # if points[17] is not None:
    #     n17 = 17
    # elif points[18] is not None:
    #     n17 = 18
    # elif points[19] is not None:
    #     n17 = 19
    # elif points[20] is not None:
    #     n17 = 20
    # else:
    #     n17 = 0
    #
    # if points[5] is not None:
    #     n5 = 5
    # elif points[6] is not None:
    #     n5 = 6
    # elif points[7] is not None:
    #     n5 = 7
    # elif points[8] is not None:
    #     n5 = 8
    # else:
    #     n5 = 0
    #
    # if points[4] is not None:
    #     n4 = 4
    # elif points[3] is not None:
    #     n4 = 3
    # else:
    #     n4 = 0
    #
    # # devo capire come e' orientata la mano, se la coordinata x e' il riferimento
    # # per dx o sx o se devo guardare la y.
    #
    # dx = abs(points[n17][0] - points[n5][0]) # range nocche su x
    # dy = abs(points[n17][1] - points[n5][1]) # range nocche su y
    # if dx > dy:
    #     # se range x > range y, coordinata di riferimento la x
    #     val = 0
    # else:
    #     val = 1
    #
    # # controllo se il pollice sta a dx o sx per capire che mano e'
    # if (points[n17][val] < points[0][val]) or (points[n5][val] > points[0][val]):
    #     # se nocca del mignolo sta a sx dello 0 e nocca dell'indice
    #     # sta a dx dello 0, allora il pollice sta a dx della nocca 5
    #     if points[n4][val] < points[n5][val]:
    #         # pollice chiuso perche' la sua x e' minore di quella dell'altro keypoint
    #         handmap.append(0)
    #     else:
    #         # altrimenti il pollice e' aperto
    #         handmap.append(1)
    # else:
    #     # altri due casi
    #     if points[n4][val] > points[n5][val]:
    #         # pollice chiuso perche' la sua x e' minore di quella dell'altro keypoint
    #         handmap.append(0)
    #     else:
    #         # altrimenti il pollice e' aperto
    #         handmap.append(1)

    # da qui inizia a calcolare le dita oltre il pollice

    #j = 5
    j = 1
    fop = []
    for k in range(0, 5): # per le 5 dita
        finger = []
        for i in range(j,j+4): # legge i keypoints delle dita
            #print('keypoint: ' + str(points[i]))
            if points[i]:
                # se non e' None
                finger.append(points[i])
            else:
                # altrimenti ci appende il valore precedente
                # praticamente duplica il keypoint (finger[-1])
                # cosi non funziona quindi ci metto il punto zero
                # che tanto nella distanza euclidea la distanza e' relativa a zero
                finger.append(points[0])
        #print(points)
        #print('finger ' + str(k) + ': ' + str(finger))
        # check sui detected keypoints per controllare quanto sono distanti
        # controlla la distanza dal punto zero: se le distanze relative di ogni
        # keypoint del dito sono simili, allora il dito e' chiuso
        # altrimenti il dito e' aperto (potrebbe comunque servire una thresh dinamica)
        distances = np.array([distance.euclidean(points[0], finger[0]),
                             distance.euclidean(points[0], finger[1]),
                             distance.euclidean(points[0], finger[2]),
                             distance.euclidean(points[0], finger[3])])
        fop.append(distances[-1])

        if (distances[-1] > 0):
            if np.any(distances==0):
                # ipotizzo il dito chiuso se manca anche solo un keypoint della sua catena
                handmap.append(0)
            # elif ((distances[-1] - distances[0])/distances[-1]) < 0.10:
            #     # closed
            #     handmap.append(0)
            elif ((distances[-1] - distances[0])/distances[-1]) < 0.10:
                handmap.append(0)
            else:
                handmap.append(1)
        else:
            # se manca proprio l'ultimo keypoint lo metto nullo
            handmap.append(0)

        j = j + 4

    # i = 5
    # finger = []
    # knuckle = []
    # for k in range(1, 5): # per le 4 dita da indice a mignolo
    #     #print('keypoint: ' + str(points[i]))
    #     # prende esclusivamente i keypoint finali del dito
    #
    #     if points[i+3]:
    #         # se non e' None
    #         finger.append(points[i+3])
    #     else:
    #         finger.append(points[0])
    #
    #     if points[i]:
    #         # se non e' None
    #         knuckle.append(points[i])
    #     else:
    #         knuckle.append(finger[-1]) #latest finger added
    #
    #     i = i + 4
    #
    # distances = np.array([distance.euclidean(points[0], finger[0]), #indice
    #                       distance.euclidean(points[0], finger[1]), #medio
    #                       distance.euclidean(points[0], finger[2]), #anulare
    #                       distance.euclidean(points[0], finger[3])]) #mignolo
    #
    # ref_distance = np.array([distance.euclidean(finger[0], knuckle[0]), #indice
    #                       distance.euclidean(finger[1], knuckle[1]), #medio
    #                       distance.euclidean(finger[2], knuckle[2]), #anulare
    #                       distance.euclidean(finger[3], knuckle[3])]) #mignolo
    #
    # print(ref_distance)
    # # check su indice
    # for i in range(0,4):
    #     if max(distances == distances[0]) and distances[0] > 0 and i == 0:
    #         # se il valore piu alto e' quello dell'indice, allora ho index aperto
    #         handmap.append(2)
    #
    #     elif ref_distance[i] > 0 and ref_distance[i] > 95:
    #         # aperto
    #         handmap.append(1)
    #     else:
    #         handmap.append(0)
    # print(handmap)

    if max(fop) == fop[1]: # se il max e' l'indice
        handmap[1] = 2

    return handmap

# ok
def gesture(points, chain, acquire, chvalue, frame):
    ''' Function to check which gesture is performed. If index only, extracts the
    coordinates, if all the fingers are present starts acquiring the index position. '''

    handmap = closed_finger(points)
    # da qui capisco che gesto e'
    # se tutti 1 -> open hand
    # se index x 1 x x x con points[8] >> degli altri

    if sum(handmap) >= 5:
        # tutti aperti
        rospy.loginfo(color.BOLD + color.GREEN + 'HAND OPEN' + color.END)
        G = 'HAND OPEN'
        acquire = True
        chain = []
    elif handmap[1] == 2: #and sum(handmap[2:4]) == 0:
        # index
        rospy.loginfo(color.BOLD + color.CYAN + 'INDEX' + color.END)
        G = 'INDEX'
        if acquire == True and len(chain) < chvalue and points[8]:
            # estrae tot volte le coordinate xy dell'indice in px, coordinata 8
            chain.append(points[8])
            #error(points[8], frame)
    else:
        rospy.loginfo(color.BOLD + color.PURPLE + 'NO GESTURE' + color.END)
        G = 'NO GESTURE'
        # non svuoto perche' accetto il disturbo durante il riempimento della coda
        # che verra' svuotata quando raggiunge il chvalue


        # da pylibfreenect2
        # prende in input il frame depth undistorted e le (i, j) del punto
        # nella matrice dell'immagine depth. i e' la riga, j la colonna
        # restituisce XYZ del punto reale corrispondente gia' in metri
        # bisognera' mappare questo dato rispetto a un certo sistema di riferimento
        # comodo, ad esempio centrato rispetto al master, cosi' da essere di facile lettura
    #    getPointXYZ(depth_frame, i, j)

        # undistorted depth frame e registered color frame sono (512, 424)
        # NOTA: verifica se per caso acquisisce un bigdepth frame, che ha la dimensione
        # del frame rgb (1920, 1080). Non so cosa e' invece color_depth_map che e' pure un np array
        # https://github.com/r9y9/pylibfreenect2/blob/master/examples/multiframe_listener.py

    return chain, acquire, G, str(handmap)

def map_workspace():
    ''' Funzione che mappa il workspace dell'operatore, che inquadra solo le mani,
    sul workspace del robot dove e' presente l'oggetto. Tutti i punti di W1 sono riferiti a W2
    quindi posso rappresentare con questa conversione semplice il punto dell'indice
    sull'altra immagine in tempo reale '''
    print('Pippo')

#######

def myhook():
    rospy.loginfo(color.BOLD + color.RED + '\n -- KEYBOARD INTERRUPT, SHUTTING DOWN --' + color.END)

def main():
    ''' Pubblica le coordinate dell'indice in tempo reale in un topic. Quando necessario
    manda le coordinate di movimento al robot. Pubblicare a video i due workspace come
    acquisizione RGB in tempo reale '''

    rospy.init_node('kinect_node')
    net = init_network()
    # input image dimensions for the network
    nPoints = 22
    kinect = Kinect(True, True, True, True, K, D)
    chain = []
    acquire = False
    threshold = 0.2
    draw = True
    chvalue = 7
    G = 'INIT'
    H = 'INIT'
    #image = cv2.imread('/home/optolab/ur2020_workspace/src/telemove/src/HandPose/hand.jpg')
    #cut = init_workspace(kinect)

    while not rospy.is_shutdown():
        kinect.acquire()
        #frame = kinect.RGBundistorted.copy()
        frame = kinect.color_new.copy()
        # per ogni frame acquisito dalla kinect deve fare sta cosa
        points, inference_time = hand_keypoints(net, frame, threshold, nPoints)

        #points, inference_time = hand_keypoints(net, image, threshold, nPoints)


        if all(x == None for x in points[1:]) or points[0] == None:
            rospy.loginfo(color.BOLD + color.RED + 'NO GESTURE FOUND IN IMAGE' + color.END)
            draw = False
            G = 'NO GESTURE'
            H = 'NONE'
        else:
            chain, acquire, G, H = gesture(points, chain, acquire, chvalue, frame)
            draw = True
            if len(chain) == chvalue:
                # ho riempito la coda
                acquire = False
                # elabora la media delle posizioni points acquisite
                # se non e' index chain points e' vuota
                #print('C: ' + str(chain))
                rechain = zip(*chain)
                mean = np.array([[sum(rechain[0])/len(rechain[0]), sum(rechain[1])/len(rechain[1]), 1.0]])
                rospy.loginfo(color.BOLD + color.YELLOW + 'CHAIN VALUE REACHED. MEAN IS: ' + str(mean[0][0]) + ' ,' + str(mean[0][1]) + color.END)
                # svuoto la coda
                chain = []
                #print(mean)
                Ref = px2meters(reference, K, R, t)
                print('Ref: ' + str(Ref))
                Ph = px2meters(mean, K, R, t)
                print('Point: ' + str(Ph))
                new = (Ph - Ref)*100 #in cm
                print(new)
                Pr = H2R(new)
                rospy.loginfo(color.BOLD + color.YELLOW + 'POINT IN H: ' + str(Ph) + ' POINT IN R: ' + str(Pr) + color.END)
                # qui converto mean nel corrispondente punto del robot e lo muovo
        draw_skeleton(frame, points, draw, inference_time, G, H)
        #draw_skeleton(image, points, draw, inference_time, G)


        # portion needed to correctly close the opencv image window
        #cv2.imshow("color", kinect.color_new)
        #cv2.imshow("depth", kinect.depth_new)
        #cv2.imshow("registered", kinect.registered_new)
        #cv2.imshow("undistorted", kinect.undistorted_new)
        #cv2.imshow("bigdepth", kinect.color_new)
        #cv2.imshow("color_depth_map", kinect.color_depth_map_new)
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break
    kinect.stop()
    rospy.on_shutdown(myhook)

if __name__ == '__main__':
    main()
