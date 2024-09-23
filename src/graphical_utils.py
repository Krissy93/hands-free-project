import cv2
import numpy as np

class Color:
    ''' Class used to print colored info on terminal.
    First call "BOLD", a color (e. g. "YELLOW")
    and at the end of the print call "END". '''

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def crop(img, pt1, pt2):
    ''' Crops the image according to starting point pt1 coordinates (x1, y1) and
    end point pt2 coordinates (x2, y2). Gets first y and then x in the crop because of
    how image arrays are defined! '''

    img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    return img

def correzione_prospettica(B3, A1, A3, b1, R, t, K, D, image):
    ''' Prende in ingresso i punti in pixel che corrispondono all'angolo in alto a sx
    dei punti B1, B3, A1, A3 in coord (x --,y ||). Questi punti corrispondono in pixel
    alle coordinate vere riconvertite (da metri a px ottengo quelle vere).
    Creo poi il cut dell'immagine eliminando la distorsione prospettica. '''

    # b1 = reference is already the px version of point (0,0) calculated by the calib matrix
    b3, _ = cv2.projectPoints(np.array([[0.0, 456.0, 0.0]]), R, t, K, D)
    a1, _ = cv2.projectPoints(np.array([[700.0, 0.0, 0.0]]), R, t, K, D)
    a3, _ = cv2.projectPoints(np.array([[700.0, 456.0, 0.0]]), R, t, K, D)

    b3 = b3.flatten()
    a1 = a1.flatten()
    a3 = a3.flatten()
    b1 = b1.flatten()
    # print('b1: ' + str(b1[0]) + ' ' + str(b1[1]))
    # print('b3: ' + str(b3[0]) + ' ' + str(b3[1]) + ' B3: ' + str(B3[0]) + ' ' + str(B3[1]))
    # print('a1: ' + str(a1[0]) + ' ' + str(a1[1]) + ' A1: ' + str(A1[0]) + ' ' + str(A1[1]))
    # print('a3: ' + str(a3[0]) + ' ' + str(a3[1]) + ' A3: ' + str(A3[0]) + ' ' + str(A3[1]))

    distX1 = np.sqrt((b1[0] - a1[0])**2 + (b1[1] - a1[1])**2)
    distX2 = np.sqrt((b3[0] - a3[0])**2 + (b3[1] - a3[1])**2)
    distY1 = np.sqrt((b1[0] - b3[0])**2 + (b1[1] - b3[1])**2)
    distY2 = np.sqrt((a1[0] - a3[0])**2 + (a1[1] - a3[1])**2)
    maxW = max(distX1, distX2)
    maxH = max(distY1, distY2)

    destination = np.array([
                  [b1[0], b1[1]], #b1
                  [maxW+b1[0], b1[1]], #a1
                  [maxW+b1[0], maxH+b1[1]], #a3
                  [b1[0], maxH+b1[1]]], dtype='float32') #b3
    # creo i punti originali
    original = np.array([
                  [b1[0], b1[1]],
                  [A1[0], A1[1]],
                  [A3[0], A3[1]],
                  [B3[0], B3[1]]], dtype='float32')

    # calcolo la matrice di correzione prospettica
    ProspM = cv2.getPerspectiveTransform(original, destination)
    # la applico all'immagine per correggerla
    warpedI = cv2.warpPerspective(image, ProspM, (image.shape[1], image.shape[0]))
    return warpedI, (b1[0], maxH+b1[1]), (maxW+b1[0], b1[1]), (maxW+b1[0], maxH+b1[1])



def draw_workspace(frame, points=[], markers=[]):
    ''' Function to draw the points corresponding to the markers in the workspace H.

    INPUTS:
    - frame: image on which workspace and markers should be drawn
    - points: corner points of workspace, they will be used to draw a polyline.
              This is a list of pixel points like: [(x1,y1),(x2,y2)]
    - markers: if it's not empty, the markers inside are expressed as pixel coordinates.
               This is a list of pixel points like: [(x1,y1),(x2,y2)]

    OUTPUTS:
    - updates frame with workspace and markers drawn on it
    '''
    # draw points on frame if markers list is not empty
    for m in markers:
        # frame, point, radius, color, thickness, linetype
        # warning: cv2 uses B-G-R format for color
        cv2.circle(frame, m, 6, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    if len(points) > 0:
        cv2.polylines(frame, [np.array(points)], True, (0, 255, 0), thickness=2)

def draw_reference(img, corners, imgpts):
    ''' Function to draw the axes of the reference system on the image frame.

    INPUTS:
    - img: image on which the axes of the reference system are drawn
    - corners: array containing the corner coordinates on which the reference system will be drawn
               this is a tuple of (xyz) coordinates acting as point (0,0,0)
    - imgpts: array containing the (xyz) coordinates of the three axes

    OUTPUTS:
    - updates img with the reference system drawn in RGB - XYZ color code
    '''

    # gets the tuple version of the corner coordinates acting as point (0,0,0)
    corner = tuple(corners[0].ravel().astype(int))
    
    # draw lines of each axis starting from point 0 to the axis ending point
    # each ending point is contained in imgpts but they must be converted to tuple
    cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3) # x red
    cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3) # y green
    cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3) # z blue

def draw_trajectory(frame, to_move):
    ''' Function to draw the actual trajectory points after filtering.

    INPUTS:
    - frame: image on which the trajectory is drawn
    - to_move: list of points in pixel coordinates that should be drawn on screen

    OUTPUT:
    - updates frame by drawing the trajectory points
    '''

    for point in to_move:
        P = (int(point[0][0]), int(point[0][1]))
        cv2.circle(frame, P, 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

#### todo mediapipe version
def draw_skeleton(frame, points, pairs):
    ''' Function to draw openpose skeleton of the hand according to a pre-defined pose pair scheme to the frame.

    INPUTS:
    - frame: image in which the skeleton will be drawn
    - points: keypoints of the hand to be drawn

    OUTPUTS:
    - updates frame with skeleton drawing
    '''

    for i in range(0, len(pairs)):
        # pose pairs represent the lines connecting two keypoints, used to correclty draw the skeleton
        partA = pairs[i][0]
        partB = pairs[i][1]

        # defines colors according to finger
        if i <= 3:
            # thumb
            colorP = (0, 0, 255)
            colorL = (21, 16, 164)
        elif i > 3 and i <= 7:
            # index
            colorP = (0, 255, 0)
            colorL = (50, 115, 12)
        elif i > 7 and i <= 11:
            # middle
            colorP = (235, 177, 17)
            colorL = (139, 103, 3)
        elif i > 11 and i <= 15:
            # annular
            colorP = (12, 200, 220)
            colorL = (10, 149, 165)
        else:
            # pinky
            colorP = (155, 101, 226)
            colorL = (78, 19, 155)

        # if there is a point in both keypoints of the pair, draws the point and the connected line
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], colorL, 3)
            if partA == 0:
                # for the reference point use black to draw the keypoint marker
                cv2.circle(frame, points[partA], 4, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
            else:
                cv2.circle(frame, points[partA], 4, colorP, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 4, colorP, thickness=-1, lineType=cv2.FILLED)

def draw_gesture_info(frame, inference_time, gesture, handmap):
    ''' Function to draw the gesture infos including the gesture detected and the inference time.

    INPUTS:
    - frame: image on which info will be drawn
    - inference_time: time needed by the network to perform skeleton detection on frame
    - gesture: last gesture detected, this may not be the same as the handmap, because the gesture
               changes according to max_chain value
    - handmap: handmap of current gesture

    OUTPUTS:
    - frame: image with text drawn on it
    '''
    # draw info on frame if the draw flag has been set as True
    #frame = cv2.putText(frame, 'INFERENCE TIME: ' + str(inference_time) + ' SEC', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    #(80, 65, 242), 3, cv2.LINE_AA)
    frame = cv2.putText(frame, 'LATEST GESTURE DETECTED: ' + str(gesture), (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (80, 65, 242), 3, cv2.LINE_AA)
    frame = cv2.putText(frame, ' CURRENT HANDMAP: ' + str(handmap), (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (80, 65, 242), 3, cv2.LINE_AA)
    return frame
