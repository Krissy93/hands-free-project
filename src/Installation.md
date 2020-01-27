# Installation instructions for Ubuntu 18

## Dependencies

1) **CUDA 10.0 and CuDNN 7.5:** install them following [this](https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73#file-cuda_10-0_installation_on_ubuntu_18-04) tutorial
2) **Caffe:** follow [this](https://caffe.berkeleyvision.org/install_apt.html) tutorial
3) **ROS:** Ubuntu 16 uses kinetic, Ubuntu 18 uses [melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
4) **OpenCV3:** this may take a while (like 40mins to 1h). Note that we only use python2 (because of ROS), so everything must be installed for python2 not python3!<br>
Open a new terminal and type:
   ```
   sudo apt-get update
   sudo apt-get upgrade
   sudo apt-get remove x264 libx264-dev
   sudo apt-get install build-essential checkinstall cmake pkg-config yasm
   sudo apt-get install git gfortran
   sudo apt-get install libjpeg8-dev libjasper-dev libpng12-dev
   sudo apt-get install libtiff5-dev
   sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
   sudo apt-get install libxine2-dev libv4l-dev
   sudo apt-get install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
   sudo apt-get install qt5-default libgtk2.0-dev libtbb-dev
   sudo apt-get install libatlas-base-dev
   sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
   sudo apt-get install libvorbis-dev libxvidcore-dev
   sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
   sudo apt-get install x264 v4l-utils
   sudo apt-get install libprotobuf-dev protobuf-compiler
   sudo apt-get install libgoogle-glog-dev libgflags-dev
   sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen 
   sudo apt install libpcl-dev
   sudo apt-get install python-dev python-pip
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
5) **libfreenect2:** needed to get frames from the Kinect v2 camera. Libfreenect2 main repository can be found [here](https://github.com/OpenKinect/libfreenect2). <br>
If using a RealSense or other camera, install their dependencies (For the RealSense, check out the Librealsense repository [here](https://github.com/IntelRealSense/librealsense). We haven't tested it but it probably works).<br>
Open a terminal and type:
    ```
    git clone https://github.com/OpenKinect/libfreenect2.git
    cd libfreenect2
    sudo apt-get install cmake pkg-config libusb-1.0-0-dev libturbojpeg0-dev libjpeg-turbo8-dev libglfw3-dev
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
6) **pylibfreenect2:** this is the python wrapper needed to correctly set up the Kinect acquisition. Official repository and installation instructions are [here](https://github.com/r9y9/pylibfreenect2/blob/master/docs/installation.rst).<br>
Basically after `libfreenect2` installation (the default installation directory is `usr/local`, so check if it is installed there) simply type:
    ```
    pip install --user Cython
    pip install --user pylibfreenect2
    ```
