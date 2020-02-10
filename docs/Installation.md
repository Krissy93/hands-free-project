# Installation instructions for Ubuntu 18

## Dependencies

- **Basic dependencies:** useful if you have a fresh install of Ubuntu. We have an Nvidia 1060 GPU so we also install the nvidia driver on our **Ubuntu 18** machine:
  ```
  sudo apt-get update
  sudo apt-get upgrade
  sudo apt-get install build-essential dkms git software-properties-common g++ freeglut3-dev libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
  ```
  For the graphic driver **on Ubuntu 18**:
  ```
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
  ```
- **CUDA 10.0 and CuDNN 7.5:** install them following [this](https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73#file-cuda_10-0_installation_on_ubuntu_18-04) tutorial:
   ```
   sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-0 cuda-drivers
   echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   sudo ldconfig
   ```
   To download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup, then download cuDNN v7.5 form https://developer.nvidia.com/cudnn.
   Be sure to be in the download folder and type:
   ```
   CUDNN_TAR_FILE="cudnn-10.0-linux-x64-v7.5.0.56"
   tar -xzvf ${CUDNN_TAR_FILE}
   ```
   Copy the extracted file in the CUDA toolkit directory:
   ```
   sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.0/include
   sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/
   sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
   ```
   Check if everything went fine by typing in a new Terminal `nvidia-smi` first and `nvcc -V` second.
- **ROS:** Ubuntu 16 uses [kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu), Ubuntu 18 uses [melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
   ```
   sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
   sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
   sudo apt update
   sudo apt install ros-melodic-desktop-full python-rosinstall python-rosinstall-generator python-wstool build-essential
   sudo rosdep init
   rosdep update
   echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   sudo apt-get install ros-melodic-cv-bridge ros-melodic-vision-opencv
   ```
- **OpenCV3:** this may take a while (like 40mins to 1h). Note that we only use python2 (because of ROS), so everything must be installed for python2 not python3!<br>
Open a new terminal and type:
   ```
   sudo apt-get update
   sudo apt-get upgrade
   sudo apt-get remove x264 libx264-dev
   sudo apt-get install checkinstall cmake pkg-config yasm git gfortran libjpeg8-dev libjasper-dev libpng12-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev libhdf5-dev doxygen libpcl-dev python-dev python-pip
   sudo -H pip2 install -U pip numpy
   pip install --user --upgrade  numpy scipy matplotlib scikit-image scikit-learn ipython
   git clone https://github.com/opencv/opencv.git
   cd opencv
   git checkout 3.3.1
   cd ..
   git clone https://github.com/opencv/opencv_contrib.git
   cd opencv_contrib
   git checkout 3.3.1
   cd ..
   cd opencv
   mkdir build
   cd build
   cmake -D CMAKE_BUILD_TYPE=RELEASE \
         -D CMAKE_INSTALL_PREFIX=/usr/local \
         -D INSTALL_C_EXAMPLES=ON \
         -D INSTALL_PYTHON_EXAMPLES=ON \
         -D WITH_TBB=ON \
         -D WITH_V4L=ON \
         -D WITH_QT=ON \
         -D WITH_OPENGL=ON \
         -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
         -D BUILD_EXAMPLES=ON ..
   ```
   Ready? Launch this and go eat something in the meantime... Note that `-j8` represents the number of CPU cores, we have 8 so we exploit them all, but you have to use as much as your system supports. Nowadays, `-j4` is usually a safe option!
   ```
   make -j8
   sudo make install
   sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
   sudo ldconfig
   ```
- **libfreenect2:** needed to get frames from the Kinect v2 camera. Libfreenect2 main repository can be found [here](https://github.com/OpenKinect/libfreenect2). <br>
If using a RealSense or other camera, install their dependencies (For the RealSense, check out the Librealsense repository [here](https://github.com/IntelRealSense/librealsense). We haven't tested it but it probably works).<br>
Open a terminal and type:
    ```
    git clone https://github.com/OpenKinect/libfreenect2.git
    cd libfreenect2
    sudo apt-get install libusb-1.0-0-dev libturbojpeg0-dev libjpeg-turbo8-dev libglfw3-dev
    mkdir build && cd build
    cmake .. -DENABLE_CXX11=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCUDA_PROPAGATE_HOST_FLAGS=off
    make
    make install
    echo '# ATTR{product}=="Kinect2"
    SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c4", MODE="0666"
    SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d8", MODE="0666"
    SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d9", MODE="0666"' > ~/90-kinect2.rules
    sudo mv ~/90-kinect2.rules /etc/udev/rules.d/90-kinect2.rules
    ```
    Connect the Kinect v2 to a USB 3.0 port. Be sure that the USB controller can manage the Kinect, since the camera usually takes up a lot of bandwidth. After this, check the installation of libfreenect by launching Protonect:
    ```
    cd build/bin
    ./Protonect
    ```
- **pylibfreenect2:** this is the python wrapper needed to correctly set up the Kinect acquisition. Official repository and installation instructions are [here](https://github.com/r9y9/pylibfreenect2/blob/master/docs/installation.rst).<br>
Basically after `libfreenect2` installation (the default installation directory is `usr/local`, so check if it is installed there) simply type:
    ```
    pip install --user Cython
    pip install --user pylibfreenect2
    ```
- **Caffe:** follow [this](https://medium.com/@atinesh/caffe-installation-on-ubuntu-18-04-lts-python-2-7-8e8c388ce51f) tutorial:
   ```
   sudo apt install python-opencv
   sudo apt-get install libatlas-base-dev
   sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
   sudo apt-get install — no-install-recommends libboost-all-dev
   sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
   sudo pip install protobuf
   sudo apt-get install the python-dev
   ```
   Go to the folder where you want to install Caffe, usually `\home`, and type:
   ```
   git clone https://github.com/BVLC/caffe.git
   cd caffe
   ```
   To install caffe we need to edit the `Makefile.config`. Copy the example:
   ```
   cp Makefile.config.example Makefile.config
   ```
   then open it and make the following edits:
   ```
   OPENCV_VERSION := 3 # if you are using OpenCV 3 or above
   CUDA_ARCH := 
   # -gencode arch=compute_20,code=sm_20 \
   # -gencode arch=compute_20,code=sm_21 \
    -gencode arch=compute_30,code=sm_30 \
    -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_61,code=compute_61
   BLAS := atlas # if you’ve installed Atlas (default)
   or
   BLAS := open # if you’ve installed OpenBLAS
   PYTHON_INCLUDE := /usr/include/python2.7 \
    /usr/lib/python2.7/dist-packages/numpy/core/include
   INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
   LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
   ```
   Save everything and close the file.
   Make sure to be in the `\caffe` root directory and install the program:
   ```
   make all -j4
   make pycaffe -j4
   ```
   Finally, add the install path to the bashrc:
   ```
   echo "export PYTHONPATH=$HOME/caffe/python:$PYTHONPATH" >> ~/.bashrc
   ```
   If everything works by opening a Terminal and calling the python interpreter, you should be able to use `import caffe` without issues!
