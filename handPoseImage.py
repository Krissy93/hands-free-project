from __future__ import division
import cv2
import time
import numpy as np
from scipy.spatial import distance

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

    # check per definire il valore dei riferimenti se qualche keypoint manca
    if points[17] is not None:
        n17 = 17
    elif points[18] is not None:
        n17 = 18
    elif points[19] is not None:
        n17 = 19
    elif points[20] is not None:
        n17 = 20
    else:
        n17 = 0

    if points[5] is not None:
        n5 = 5
    elif points[6] is not None:
        n5 = 6
    elif points[7] is not None:
        n5 = 7
    elif points[8] is not None:
        n5 = 8
    else:
        n5 = 0

    if points[4] is not None:
        n4 = 4
    elif points[3] is not None:
        n4 = 3
    else:
        n4 = 0

    # devo capire come e' orientata la mano, se la coordinata x e' il riferimento
    # per dx o sx o se devo guardare la y.

    dx = abs(points[n17][0] - points[n5][0]) # range nocche su x
    dy = abs(points[n17][1] - points[n5][1]) # range nocche su y
    if dx > dy:
        # se range x > range y, coordinata di riferimento la x
        val = 0
    else:
        val = 1

    print(val)
    print(points[n17])
    print(points[n5])
    print(points[0])

    # controllo se il pollice sta a dx o sx per capire che mano e'
    if (points[n17][val] < points[0][val]) or (points[n5][val] > points[0][val]):
        # se nocca del mignolo sta a sx dello 0 e nocca dell'indice
        # sta a dx dello 0, allora il pollice sta a dx della nocca 5
        if points[n4][val] < points[n5][val]:
            # pollice chiuso perche' la sua x e' minore di quella dell'altro keypoint
            handmap.append(0)
        else:
            # altrimenti il pollice e' aperto
            handmap.append(1)
    else:
        # altri due casi
        if points[n4][val] > points[n5][val]:
            # pollice chiuso perche' la sua x e' minore di quella dell'altro keypoint
            handmap.append(0)
        else:
            # altrimenti il pollice e' aperto
            handmap.append(1)

    # da qui inizia a calcolare le dita oltre il pollice

    j = 5
    for k in range(1, 5): # per le 4 dita pollice escluso
        finger = []
        for i in range(j,j+4): # legge i keypoints delle dita
            if points[i]:
                # se non e' None
                finger.append(points[i])
            else:
                # altrimenti ci appende il valore precedente
                # praticamente duplica il keypoint
                finger.append(finger[-1])

        # check sui detected keypoints per controllare quanto sono distanti
        # controlla la distanza dal punto zero: se le distanze relative di ogni
        # keypoint del dito sono simili, allora il dito e' chiuso
        # altrimenti il dito e' aperto (potrebbe comunque servire una thresh dinamica)
        distances = np.array([distance.euclidean(points[0], finger[0]),
                             distance.euclidean(points[0], finger[1]),
                             distance.euclidean(points[0], finger[2]),
                             distance.euclidean(points[0], finger[3])])
        '''distances = np.array([distance.euclidean(finger[0], finger[1]),
                             distance.euclidean(finger[1], finger[2]),
                             distance.euclidean(finger[2], finger[3])])'''
        '''dx = np.array([x[0] for x in finger])
        dy = np.array([x[1] for x in finger])
        print(np.amax(dx) - np.amin(dx))
        print(np.amax(dy) - np.amin(dy))'''

        # WARNING: quanto e' robusto rispetto allo zoom?
        if ((distances[-1] - distances[0])/distances[-1]) < 0.20:
            # closed
            handmap.append(0)
        else:
            handmap.append(1)

        j = j + 4

    return handmap

def gesture(points):
    ''' Function to check which gesture is performed. If index only, extracts the
    coordinates, if all the fingers are present starts acquiring the index position. '''

    # if the network recognizes a keypoint in all 5 fingers (only last pair of each
    # finger is checked), then the gesture recognized is HAND OPEN

    handmap = closed_finger(points)
    print(handmap)
    # da qui capisco che gesto e'
    # se tutti 1 -> open hand
    # se index x 1 0 0 0 -> estraggo l'index coordinate

    if sum(handmap) == 5:
        # tutti uno
        print('HAND OPEN')
    elif handmap[1] == 1 and sum(handmap[2:-1]) == 0:
        # index
        print('INDEX')
    else:
        print('NO GESTURE')

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#frame = cv2.imread("right-frontal.jpg")
frame = cv2.imread("pointing.jpg")
#original_frame = cv2.imread("front-back.jpg") # ci sono due mani, devo dividere il frame in due immagini distinte!
frameCopy = np.copy(frame)
#frame = original_frame[:, 0:620] #img[y:y+h, x:x+w]
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
aspect_ratio = frameWidth/frameHeight

threshold = 0.1

t = time.time()
# input image dimensions for the network
inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

# Empty list to store the detected keypoints
points = []
probabilitymap = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    probabilitymap.append(prob)

    if prob > threshold :
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    else :
        points.append(None)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)


cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))
gesture(points)

cv2.waitKey(0)
