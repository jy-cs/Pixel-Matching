# Camera Calibration
This project consists of two parts:
## Part I
We use the OpenCV library to write a calibration program that takes as input a list of 3D coordinates with their corresponding 2D pixel coordinates and outputs the 3x4 calibration matrix on the screen. In addition, our program should also print the average errors (difference between projected points and original pixels).

## Part II
We write our own program to calibrate an RGB camera using the SVD routine.
Our program takes as input a list of 3D coordinates with their corresponding 2D pixel coordinates and outputs the 3x4 calibration matrix on the screen. Similarly, the program also print the average errors (difference between projected points and original pixels).

## Useful resources:
[OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
[OpenCV: Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

### Opencv on google colab:
[Getting Started with OpenCV - Colaboratory
[OpenCV installation Google Colab - Stack Overflow
### Linear Algebra Python:
[numpy.linalg.svd
[Gram–Schmidt process
[numpy.linalg.norm

### Python Fundamentals:
[How to open two files together in Python? - GeeksforGeeks
[numpy.array — NumPy v1.24 Manual
[How to Read a Text file In Python Effectively
[numpy.matmul — NumPy v1.24 Manual
[numpy.append — NumPy v1.24 Manual
[numpy.transpose — NumPy v1.24 Manual
[Indexing — NumPy v1.10 Manual
[numpy.identity — NumPy v1.24 Manual
[What is the meaning of "int(a\[::-1\])" in Python? - Stack Overflow
[\[::-1\] in Python with Examples


### OpenCV:
[rvecs](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html)
[how can i get the camera projection matrix out of calibrateCamera() return values - Stack Overflow
[Why does OpenCV think I am using a non-planar calibration rig? - Stack Overflow
[Camera calibration using multiple images - Signal Processing Stack Exchange
[How to get camera focal length with OpenCV - Stack Overflow
[Camera calibration With OpenCV
[Perspective-n-Point (PnP) pose computation
[solvePnPRefineVVS() Camera Calibration and 3D Reconstruction

### Camera Calibration:
[Projector-Camera System Calibration and Non-planar Scene Estimation
[A Flexible New Technique for Camera Calibration
[Camera calibration With OpenCV
[How to get camera focal length with OpenCV - Stack Overflow

### Transformation OpenCV:
[Perspective Transformation - Python OpenCV - GeeksforGeeks
[OpenCV: solve() Operations on arrays
[CPnP:Consistent-Pose-Estimator-for-Perspective-n-Point-Problem](https://arxiv.org/pdf/2209.05824.pdf)

### Lec Notes:
https://www.cse.unr.edu/~bebis/CS791E/Notes/CameraCalibration.pdf
http://16385.courses.cs.cmu.edu/fall2022/lecture/cameras
https://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf
Find eigenvectors and eigenvalues of (A^T)A instead of A (differences?): http://www.cs.cmu.edu/~16385/s15/lectures/Lecture17.pdf
