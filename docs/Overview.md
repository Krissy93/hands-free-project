# Overview

The project idea is to use the operator hands to teleoperate a robot end-effector. To do so, two main aspects must be defined:
- how to extract the operator hand position from the real-time image;
- how to link the extracted position to a robot end-effector position.

As a first approach, we extract the hand position using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), which detects the hand in the image frame and estimates the hand skeleton.

Robots may be Industrial robots or Co-Bots, thus the operative workspace may be a different one (even far away, because the Industrial robot is behind a safety a cage) or the same one (in the case of Co-Bots, which are safe for humans).
Considering the case of two workspaces, we have:
- the user workspace (small, controlled background to improve robustness);
- the robot workspace (big).
A point in the user workspace corresponds to a point in the robot workspace thanks to a correspondence function which converts the coordinates (Fig. 1). This step is basically a calibration between reference systems.

<p align="center">
  <img width="800" src="https://github.com/Krissy93/hands-free-project/blob/master/images/horizcalib.png">
</p>

## Workspaces Calibration
The math behind the procedure is detailed in our paper. Here a summarized description of the procedure is reported.

### User Workspace Calibration
The user workspace must be calibrated with respect to the camera (Fig. 1 green horizontal panel). To do so, standard procedures are usually applied, involving a calibration master (which has known dimensions of the patter, e. g. the chessboard squares).

To summarize the procedure, here are the required steps:
- glue the calibration master on a rigid pane
- start an acquisition procedure that saves a given number of snapshots from the camera (30 is a good value)
- move the pane around the Field of View of the camera, tilting/changing the positioning in the space; be sure to acquire mostly not-blurred images
These images are the calibration images needed for the calibration procedure.
Our script `calibrate.py` loads these images and, using the standard procedure detailed [here](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html), finds out the intrinsic parameters of the camera (named **Camera Matrix K** and **Distortion Parameters D**).

Finally, the calibration master has to be put in place in the user workspace considering its zero position. This is the reference point of reference system **H** in Fig. 1!
The script then shots a photo of the current workspace and calculates the Rotation matrix **R** and the translation vector **t** of the reference system. It also finds the corresponding undistorted **Rd** and **td**.
All the obtained values are stored in YAML file that is loaded at runtime by the main software.

### Robot Workspace Calibration
To calibrate the robot workspace, it is necessary to find, for each point of workspace **W**, the exact match between workspace **W** positions and the robot reference system **R**.
We used a centering tool that we designed and 3D printed in our Lab, customized for our robot (Sawyer robot). We placed the tool in each marker position of workspace **W** and moved the end-effector inside the tailor-made cavity of the tool. Only when the end-effector was completely inside the cavity and centered, we saved the corresponding robot pose (Fig. 2).

<p align="center">
  <img height="400" src="https://github.com/Krissy93/hands-free-project/blob/master/images/robotcalibration.png">
</p>

We repeated this procedure for all the 13 markers of the workspace. The more the points, the more accurate the rototranslation matrix would be!
Our script `calibrate_vertical_workspace.py` is used to acquire the robot positions (it reads them from the robot encoder calling the corresponding `\joint_states` ROS topic). The script saves them and the correspoding marker positions with respect to reference system **W** (zero position in the bottom-left marker of the template) in a YAML file. Then, it calculates the rototranslation matrix and saves it in another YAML file.

## Hand position extraction
To obtain a robot position, the index position of the user hand (left or right) is extracted from a frame representing the user workspace. The pixel coordinates are converted in meters and mapped to the robot workspace using the correspondence function (which is based on the camera calibration performed in advance!).

Two gestures are defined by the software (Fig. 3):
- hand open
- index only
Users can move the hands freely in the user workspace. When they want to move the robot to a specific position, they have to perform the "hand open" gesture first and the "index" gesture afterwards, keeping the latter gesture still for a certain amount of time.

<p align="center">
  <img width="600" src="https://github.com/Krissy93/hands-free-project/blob/master/images/keypoints.png">
</p>

The index coordinates are extracted as the mean of the consecutive index positions acquired. The number of consecutive index gestures acquired are defined by the parameter **chvalue**: an higher number means more robustness but also more time required to properly fill the chain of consecutive gestures.
