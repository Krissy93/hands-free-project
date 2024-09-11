import cv2
import numpy as np
import graphical_utils as gu
import time
import mediapipe as mp
import math
from scipy.spatial import distance
import rospy

class Hand:
    def __init__(self, net_params=[1], threshold=0.5, max_chain=0, debug=False):
        self.debug = debug
        self.threshold = threshold
        self.model_complexity = net_params[0]
        self.inference_time = 0
        self.max_chain = max_chain

        # Initialize the Mediapipe Hands model once
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=self.model_complexity, 
                                         min_detection_confidence=self.threshold, 
                                         min_tracking_confidence=self.threshold)
        self.mp_draw = mp.solutions.drawing_utils  # If drawing the hands is needed

        # Values for gesture chains
        self.points = []
        self.prob_list = []
        self.current_gesture = None
        self.chain_open = 0
        self.chain_move = 0
        self.acquire = False
        self.index_positions = []
        self.positions_saved = []

        # Initialize the hand connections in Mediapipe
        self.init_mediapipe()

    def init_mediapipe(self):
        """Initialize the connections between key points of the hands in Mediapipe."""
        rospy.loginfo(gu.Color.BOLD + gu.Color.YELLOW + 'INITIALIZING MEDIAPIPE NETWORK...' + gu.Color.END)
        self.pairs = []
        for i in self.mp_hands.HAND_CONNECTIONS:
            self.pairs.append(i)

    def mediapipe_inference(self, frame, n_keypoints=21):
        """Run inference on the hand key points using Mediapipe."""
        # Record the start time for inference
        before = time.time()

        # Optimize the frame for inference by disabling write permissions
        frame.flags.writeable = False
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference using the pre-initialized Mediapipe hands model
        self.output = self.hands.process(frame_rgb)

        # Calculate the inference time
        self.inference_time = round(time.time() - before, 3)

        # Empty list to store detected key points
        self.points = []
        if self.output.multi_hand_landmarks:
            for i in range(n_keypoints):
                # Convert normalized key points to pixel coordinates
                self.points.append((int(self.output.multi_hand_landmarks[0].landmark[i].x * frame.shape[1]),
                                    int(self.output.multi_hand_landmarks[0].landmark[i].y * frame.shape[0])))
        else:
            self.points = [None] * n_keypoints

    def get_handmap(self):
        """Function to check if the finger is closed or not according to the position of the detected key points.
        If their position is relatively close to the reference key point 0 (calculated as Euclidean distance),
        then the finger is closed, otherwise it is open. Returns a map of closed fingers, 
        where position 0 is the thumb and position 4 is the pinky.
        
        Handmap values:
        0: CLOSED FINGER
        1: OPEN FINGER
        2: INDEX FINGER OPEN AND SUPERIMPOSED OVER THE OTHERS
        """
        self.handmap = []
        j = 1
        fop = []

        for k in range(5):
            finger = []
            for i in range(j, j + 4):
                if self.points[i]:
                    finger.append(self.points[i])
                else:
                    finger.append(self.points[0])

            distances = np.array([distance.euclidean(self.points[0], finger[0]),
                                  distance.euclidean(self.points[0], finger[1]),
                                  distance.euclidean(self.points[0], finger[2]),
                                  distance.euclidean(self.points[0], finger[3])])
            
            fop.append(distances[-1])

            if distances[-1] > 0:
                if np.any(distances == 0):
                    self.handmap.append(0)
                elif ((distances[-1] - distances[0]) / distances[-1]) < 0.10:
                    self.handmap.append(0)
                else:
                    self.handmap.append(1)
            else:
                self.handmap.append(0)

            j += 4

        if max(fop) == fop[1]:
            self.handmap[1] = 2
        elif max(fop) == fop[2]:
            self.handmap[2] = 2

    def get_gesture(self):
        """Function to check which gesture is performed.
        Defined gestures:
        HAND OPEN: handmap = [1,1,1,1,1], all fingers open
        INDEX: handmap = [x,1,0,0,0], only index finger open
        MOVE: handmap = [x,1,1,0,0], index and middle fingers open
        NO GESTURE: none of the above
        """
        self.get_handmap()

        if self.handmap.count(1) == len(self.handmap):
            rospy.loginfo(gu.Color.BOLD + gu.Color.GREEN + 'HAND OPEN' + gu.Color.END)
            self.chain_open += 1
            self.chain_move = 0

            if self.chain_open == self.max_chain:
                self.current_gesture = 'HAND OPEN'

        elif self.handmap[1] >= 1 and self.handmap[2] >= 1:
            rospy.loginfo(gu.Color.BOLD + gu.Color.PURPLE + 'MOVE' + gu.Color.END)
            self.chain_move += 1
            self.chain_open = 0

            if self.acquire and self.chain_move == self.max_chain:
                self.current_gesture = 'MOVE'

        elif self.handmap[1] == 2 and self.handmap[2] < 1:
            rospy.loginfo(gu.Color.BOLD + gu.Color.CYAN + 'INDEX' + gu.Color.END)
            self.chain_open = 0
            self.chain_move = 0

            if self.acquire and len(self.index_positions) < self.max_chain and self.points[8]:
                self.index_positions.append(self.points[8])

                if len(self.index_positions) == self.max_chain:
                    rechain = list(zip(*self.index_positions))
                    mean = np.array([[sum(rechain[0]) / len(rechain[0]), sum(rechain[1]) / len(rechain[1]), 1.0]])

                    if len(self.positions_saved) != 0:
                        last = self.positions_saved[-1]
                        dist = math.sqrt((last[0][0] - mean[0][0]) ** 2 + (last[0][1] - mean[0][1]) ** 2)
                        if dist > 5:
                            self.positions_saved.append(mean)
                    else:
                        self.positions_saved.append(mean)

                    rospy.loginfo(gu.Color.BOLD + gu.Color.YELLOW + 'CHAIN VALUE REACHED. MEAN IS: ' + str(mean[0][0]) + ', ' + str(mean[0][1]) + gu.Color.END)
                    self.index_positions = []
                    self.current_gesture = 'INDEX'

        else:
            if self.debug:
                rospy.loginfo(gu.Color.BOLD + gu.Color.RED + 'NO GESTURE DETECTED' + gu.Color.END)
            self.current_gesture = 'NO GESTURE'
