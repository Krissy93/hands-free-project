import cv2
import numpy as np
import graphical_utils as gu
import time
import mediapipe as mp
import math
from scipy.spatial import distance
import rospy

class Hand:
    def __init__(self, net_params, threshold, max_chain=0, debug=False):
        self.debug = debug
        self.threshold = threshold
        self.model_complexity = net_params[0]
        self.inference_time = 0
        self.max_chain = max_chain

        self.mp_hands = mp.solutions.hands

        self.mpDraw = mp.solutions.drawing_utils
        self.init_mediapipe()

        # Values for gesture chains
        self.points = []
        self.prob_list = []
        self.current_gesture = None
        self.chain_open = 0
        self.chain_move = 0
        self.acquire = False
        self.index_positions = []
        self.positions_saved = []

    def init_mediapipe(self):
        rospy.loginfo(gu.Color.BOLD + gu.Color.YELLOW + 'INITIALIZING MEDIAPIPE NETWORK...' + gu.Color.END)
        self.pairs = []
        for i in self.mp_hands.HAND_CONNECTIONS:
            self.pairs.append(i)

    
    def mediapipe_inference(self, frame, n_keypoints=21):
        # Assicurati che frame sia un array di tipo uint8
        frame = np.asarray(frame, dtype=np.uint8)
        
        # Crea una copia profonda dell'immagine per le operazioni di disegno
        writable_frame = np.copy(frame)
        
        # Verifica che writable_frame sia scrivibile
        if not writable_frame.flags.writeable:
            rospy.logwarn("Writable frame is not writable!")

        # Calcola il tempo totale di inferenza come differenza tra l'inizio di questa funzione
        # e il tempo corrispondente alla creazione dell'output
        before = time.time()

        # L'immagine non è scrivibile per migliorare le prestazioni durante l'inferenza
        frame_rgb = cv2.cvtColor(writable_frame, cv2.COLOR_BGR2RGB)

        self.hands = self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.threshold, min_tracking_confidence=self.threshold)

        # Effettua l'inferenza
        # model_complexity, min_detection_confidence, min_tracking_confidence
        with self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.threshold, min_tracking_confidence=self.threshold) as hands:
            self.output = hands.process(frame_rgb)

            # Ottieni il tempo di inferenza richiesto dalla rete per eseguire il rilevamento
            self.inference_time = round(time.time() - before, 3)

            # Liste vuote per memorizzare i punti chiave rilevati e i punteggi di confidenza
            self.points = []
            if self.output.multi_hand_landmarks:
                for i in range(n_keypoints):
                    # Le coordinate sono normalizzate da 0 a 1
                    # Quindi dobbiamo trasformarle di nuovo in coordinate dell'immagine
                    self.points.append((int(self.output.multi_hand_landmarks[0].landmark[i].x * writable_frame.shape[1]), int(self.output.multi_hand_landmarks[0].landmark[i].y * writable_frame.shape[0])))

                # Disegna i landmark e le connessioni
                for pair in self.pairs:
                    x1, y1 = self.points[pair[0]]
                    x2, y2 = self.points[pair[1]]
                    cv2.line(writable_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for point in self.points:
                    cv2.circle(writable_frame, point, 5, (255, 0, 0), -1)

            else:
                self.points = [None] * n_keypoints

        return writable_frame
    '''    

    def mediapipe_inference(self, frame, n_keypoints=21):
    
        # calculate the total inference time as a difference between the starting time
        # of this function and the time corresponding to the output creation
        before = time.time()

        # image is not writable to improve performance during inference
        frame.flags.writeable = False
        # converts to rgb from bgr (standard opencv format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.hands = self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.threshold, min_tracking_confidence=self.threshold)

        # performs inference
        # model_complexity, min_detection_confidence, min_tracking_confidence
        with self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.threshold, min_tracking_confidence=self.threshold) as hands:
            self.output = hands.process(frame)
            
            # gets inference time required by network to perform the detection
            self.inference_time = round(time.time() - before, 3)

            # empty lists to store the detected keypoints and confidence scores
            self.points = []
            if self.output.multi_hand_landmarks:
                for i in range(n_keypoints):
                    # coordinates are normalized 0-1
                    # so we need to transform them back to image coordinates
                    # shape 1 is width, shape 0 is height
                    self.points.append((int(self.output.multi_hand_landmarks[0].landmark[i].x * frame.shape[1]), int(self.output.multi_hand_landmarks[0].landmark[i].y * frame.shape[0])))
            else:
                self.points = [None] * n_keypoints
    '''

    def get_handmap(self):
        ''' Function to check if the finger is closed or not, according to the position
        of the detected keypoints. If their position is relatively close to the reference
        keypoint 0 (calculated as euclidean distance), then the finger is closed, otherwise is open.
        Returns a map of fingers closed, where in position 0 there is the thumb and in position 4 the pinky.
        Handmap values are:
        0: CLOSED FINGER
        1: OPENED FINGER
        2: INDEX OPENED AND SUPERIMPOSED OVER THE OTHERS

        INPUTS:
        - self.points: list of pixel keypoints detected from network, length is n_keypoints
                  and points with confidence less than threshold have been substituted by None

        OUTPUTS:
        - updates value of self.handmap, which is a list of 5 elements corresponding to
          hand fingers from thumb to pinky. Values are 0 for a closed finger, 1 for an opened one
        '''


        # empty handmap handle
        self.handmap = []
        j = 1

        # fingertips handle
        fop = []

        # for the 5 fingers (k) we read the finger kypoints (j)
        for k in range(5):
            # empty finger handle
            finger = []
            # read finger keypoints, always 4 keypoints
            # i is the counter for points, j is the counter for the keypoint number,
            # which changes according to the current finger
            for i in range(j, j + 4):
                if self.points[i]:
                    # if the current point i exists, appends it
                    finger.append(self.points[i])
                else:
                    # else, appends keypoint 0
                    # this is okay because the relative distance of each keypoint of the finger
                    # calculated afterwards is relative to keypoint 0
                    finger.append(self.points[0])

            # calculates relative distance of each keypoint of the finger relative to
            # keypoint 0. This is needed to find if the keypoints are collapsed or not
            distances = np.array([distance.euclidean(self.points[0], finger[0]),
                                  distance.euclidean(self.points[0], finger[1]),
                                  distance.euclidean(self.points[0], finger[2]),
                                  distance.euclidean(self.points[0], finger[3])])
            
            # appends to fop handle the last value of distances, corresponding to
            # the fingertip distance from keypoint 0
            fop.append(distances[-1])

            # check if the current fingertip has a relative distance > 0 (not collapsed)
            if distances[-1] > 0:
                # then, if any keypoint of the current finger is absent, set the whole finger as closed
                if np.any(distances == 0):
                    self.handmap.append(0)
                
                # calculates the proportional "length" of the finger as:
                # first value of distances (distance from the first keypoint
                # of the finger and keypoint 0) and the latest value of distances
                # (distance from the fingertip keypoint and keypoint 0), divided by
                # the fingerip distance. If this proportion is lower than 10%,
                # the finger is set as closed
                elif ((distances[-1] - distances[0]) / distances[-1]) < 0.10:
                    self.handmap.append(0)
                # if none of the above, the finger is open!
                else:
                    self.handmap.append(1)
            # if the fingertip distance is not > 0 it means that the fingertip keypoint
            # is absent, thus we set the finger as closed
            else:
                self.handmap.append(0)
            # increment the keypoint values for the next finger
            j += 4

        # Add debug logs
        rospy.loginfo(f"Handmap before index adjustment: {self.handmap}")
        rospy.loginfo(f"FOP: {fop}")
            
        # FOR ENDED! All fingers calculated by now!
        # check the fingertips distances: if the maximum distance in the list
        # corresponds to the index finger, then it means that the index was the only
        # finger open/detected as open, thus we impose the index value in the handmap as 2
        if all(val == 1 for val in self.handmap):
            self.handmap = [1, 1, 1, 1, 1]
            rospy.loginfo("Detected hand open gesture")
        else:
            # FOR ENDED! All fingers calculated by now!
            if max(fop) == fop[1]:
                self.handmap[1] = 2
            elif max(fop) == fop[2]:
                self.handmap[2] = 2

    def get_gesture(self):
        ''' Function to check which gesture is performed.
        The idea is to define a list of fingers starting from thumb to pinky, where
        0 means closed and 1 means opened. By calling the closed_finger function
        it is possible to extract this list called "handmap" and therefore define
        the actions to perform.

        Gestures defined:
        HAND OPEN: corresponds to handmap = [1,1,1,1,1], all fingers opened
        INDEX: corresponds to handmap = [x,1,0,0,0], only index finger opened
        MOVE: corresponds to handmap = [x,1,1,0,0], index and middle fingers opened
        NO GESTURE: none of the above

        SELF INPUTS:
        - self.points: list of keypoints detected by inference function
        - self.chain_open: counter of how many times the gesture HAND OPEN has been found
        - self.chain_move: counter of how many times the gesture MOVE has been found
        - self.acquire: boolean flag defining if index positions must be acquired or not
        - self.current_gesture: string defining the actual gesture recognized when
                                a specific counter/condition has been reached. According to
                                this string, some robot ACTIONS may be performed
        - self.index_positions: list of index px points acquired consecutively
        - self.positions_saved: filtered version of self.index_positions in which points
                                too close to each other have been discarded

        SELF OUTPUTS:
        - updates values of inputs according to current handmap detected
        '''

        # obtains the current handmap
        self.get_handmap()
        rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + f'Handmap: {self.handmap}' + gu.Color.END)

        # if all values of handmap are equal to 1, then the gesture is HAND OPEN
        if self.handmap.count(1) == len(self.handmap):
            #rospy.loginfo(gu.Color.BOLD + gu.Color.GREEN + 'HAND OPEN' + gu.Color.END)
            # add one to chain hand open, resets chain move
            self.chain_open += 1
            self.chain_move = 0

            # if max chain is reached, performs action linked to gesture
            if self.chain_open == self.max_chain:
                self.current_gesture = 'HAND OPEN'
                # ACTION is performed in main script accordingly
        
        # if the index and middle finger value are equal to 1, then the gesture is MOVE
        #== [0, 1, 1, 0, 0]
        elif self.handmap[1] >= 1 and self.handmap[2] >= 1:
            #rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'MOVE' + gu.Color.END)
            # adds one to chain move, resets chain hand open
            self.chain_move += 1
            self.chain_open = 0

            # if max chain is reached and acquire is set to true, performs action linked to gesture
            if self.acquire and self.chain_move == self.max_chain:
                self.current_gesture = 'MOVE'
                # ACTION is performed in main script accordingly
        
        # if only the index finger value is equal to 1, then the gesture is INDEX
        #elif handmap == [0, 1, 0, 0, 0]:      
        elif self.handmap[1] == 2 and self.handmap[2] < 1:
            #rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + 'INDEX' + gu.Color.END)
            # resets counters for chain hand open and move
            self.chain_open = 0
            self.chain_move = 0

            # if the acquire flag has been set to True, the length of the chain
            # is less than the defined chain value and the index finger keypoint exists,
            # then it appends the index finger keypoint coordinates to the chain
            if self.acquire and len(self.index_positions) < self.max_chain and self.points[8]:
                self.index_positions.append(self.points[8])

                if len(self.index_positions) == self.max_chain:
                    # zips x coordinates in one sublist and y coordinats in another sublist
                    rechain = list(zip(*self.index_positions))
                    # calculate the mean of the saved set of index points
                    # mean is actually the point, because we acquire max_value points and
                    # perform a mean to filter out detection errors
                    mean = np.array([[sum(rechain[0]) / len(rechain[0]), sum(rechain[1]) / len(rechain[1]), 1.0]])

                    # if self.positions_saved is not empty it means other index points have been saved before

                    # for example to build a trajectory we need more than one point
                    if len(self.positions_saved) != 0:
                        last = self.positions_saved[-1]
                        dist = math.sqrt((last[0][0] - mean[0][0]) ** 2 + (last[0][1] - mean[0][1]) ** 2)
                        # do not save the point if it is too close to the preceding one
                        if dist > 5:
                            self.positions_saved.append(mean)
                    
                    # if self.positions_saved is empty I don't need to check the points mutual distance
                    # so I just save the point in the list
                    else:
                        self.positions_saved.append(mean)

                    rospy.loginfo(gu.Color.BOLD + gu.Color.YELLOW + 'CHAIN VALUE REACHED. MEAN IS: ' + str(mean[0][0]) + ', ' + str(mean[0][1]) + gu.Color.END)
                    self.index_positions = []
                    self.current_gesture = 'INDEX'

        else:
            # no gesture was detected

            # we accept the possible noises of the detection, thus we do not empty
            # the chain queue when no gesture is detected. A stronger constraint could be
            # to empty the chain in this case, and only accept values if the chain is not broken
            if self.debug:
                rospy.loginfo(gu.Color.BOLD + gu.Color.RED + 'NO GESTURE DETECTED' + gu.Color.END)
            self.current_gesture = 'NO GESTURE'


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 16:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 20:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList
