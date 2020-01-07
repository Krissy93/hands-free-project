# Hands-Free Project
The Hands-Free Project allow users to move a robot in a desired position of its workspace simply by using their hands to point to the desired positions of a simulated workspace.
To do so, it is required by the software to set up two workspaces:
- the user workspace (small, controlled background to improve robustness)
- the robot workspace (big)
A point in the user workspace corresponds to a point in the robot workspace thanks to a correspondence function which converts the coordinates. This step is basically a calibration between reference systems.

# How it works
To obtain a robot position, the index position of the user hand (left or right) is extracted from a frame representing the user workspace. The pixel coordinates are converted in meters and mapped to the robot workspace using the correspondence function.

Two gestures are defined by the software:
- hand open
- index only
Users can move the hands freely in the user workspace. When they want to move the robot to a specific position, they have to perform the "hand open" gesture first and the "index" gesture afterwards, keeping the latter gesture still for a certain amount of time.

The index coordinates are extracted as the mean of the consecutive index positions acquired. The number of consecutive index gestures acquired are defined by the parameter chvalue: an higher number means more robustness but also more time required to properly fill the chain of consecutive gestures.

# To do
- [x] Get the hand skeleton from frames in real time using Kinect v2
- [x] Define gestures (orientation and zoom invariant)
- [ ] Map the experimental workspaces
- [ ] Define a procedure to automatically map the workspaces
- [ ] Write visualization code of the two simultaneous workspaces, skeleton mapped to the robot one
