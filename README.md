# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

![img1](performance_results/Final%20Results%20:%20TTC_screenshot_13.04.2020.png)

##### Complete Pipeline of 3D Object Tracking with 3D Lidar and 2D Camera Data Fused


![img2](images/course_code_structure.png)

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)


## Build OpenCV 4.2.0 from Source

```sh
$ mkdir opencv4 && cd opencv4

# The OpenCV 4.2.0 source code can be downloaded

$ git clone https://github.com/opencv/opencv/tree/4.2.0

# The OpenCV 4.2.0 Repository for OpenCV's extra modules and Non-free algorithms

$ git clone https://github.com/opencv/opencv_contrib/tree/4.2.0


$ cd opencv && mkdir build


$ cmake   .. \
	      -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DINSTALL_C_EXAMPLES=ON \
        -DWITH_TBB=ON \
	      -DWITH_V4L=ON \
	      -DOPENCV_ENABLE_NONFREE=ON\
        -DWITH_QT=ON \
        -DWITH_OPENGL=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DBUILD_EXAMPLES=ON

$ make -j7 # runs 7 jobs in parallel

# [Optional] build the documentation
$ cd ~/opencv/build/doc/
$ make -j7 doxygen

$ cd ~/opencv/build

$ sudo make install

```

## Basic Build Instructions

1. Clone this repo.
2. Download Yolov3 Weights and extract it `cd dat/yolo && wget "https://pjreddie.com/media/files/yolov3.weights"`
1. Make a build directory in the top level project directory: `mkdir build && cd build`
2. Compile: `cmake .. && make`
3. Run it: `./3D_object_tracking`.


# 3D Object Tracking Technical 
[Tasks Achievements Report](Technical.md)

