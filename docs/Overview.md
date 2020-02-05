# Overview

The project idea is to use the operator hands to teleoperate a robot end-effector. To do so, two main aspects must be defined:
- how to extract the operator hand position from the real-time image;
- how to link the extracted position to a robot end-effector position.

As a first approach, we extract the hand position using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), which detects the hand in the image frame and estimates the hand skeleton. 
############# qui
To do so, it is required by the software to set up two workspaces:
- the user workspace (small, controlled background to improve robustness)
- the robot workspace (big)
A point in the user workspace corresponds to a point in the robot workspace thanks to a correspondence function which converts the coordinates. This step is basically a calibration between reference systems.

## Hand position extraction
To obtain a robot position, the index position of the user hand (left or right) is extracted from a frame representing the user workspace. The pixel coordinates are converted in meters and mapped to the robot workspace using the correspondence function.

Two gestures are defined by the software:
- hand open
- index only
Users can move the hands freely in the user workspace. When they want to move the robot to a specific position, they have to perform the "hand open" gesture first and the "index" gesture afterwards, keeping the latter gesture still for a certain amount of time.

The index coordinates are extracted as the mean of the consecutive index positions acquired. The number of consecutive index gestures acquired are defined by the parameter chvalue: an higher number means more robustness but also more time required to properly fill the chain of consecutive gestures.
