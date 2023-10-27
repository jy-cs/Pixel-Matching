# Camera Calibration
This project consists of two parts:
## Part I
We use the OpenCV library to write a calibration program that takes as input a list of 3D coordinates with their corresponding 2D pixel coordinates and outputs the 3x4 calibration matrix on the screen. In addition, our program should also print the average errors (difference between projected points and original pixels).

## Part II
We write our own program to calibrate an RGB camera using the SVD routine.
Our program takes as input a list of 3D coordinates with their corresponding 2D pixel coordinates and outputs the 3x4 calibration matrix on the screen. Similarly, the program also print the average errors (difference between projected points and original pixels).

## Useful resources:
* [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
* [OpenCV: Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

### Opencv on google colab:
* [Getting Started with OpenCV - Colaboratory](https://colab.research.google.com/github/farrokhkarimi/OpenCV/blob/master/Getting_Started_with_OpenCV.ipynb)
* [OpenCV installation Google Colab - Stack Overflow](https://stackoverflow.com/questions/48420659/opencv-installation-google-colab)

### Linear Algebra Python:
* [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
* [Gram–Schmidt process](https://www.section.io/engineering-education/singular-value-decomposition-in-python/)
* [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

### Python Fundamentals:
* [How to open two files together in Python? - GeeksforGeeks](https://www.geeksforgeeks.org/how-to-open-two-files-together-in-python/)
* [numpy.array — NumPy v1.24 Manual](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
* [How to Read a Text file In Python Effectively](https://www.pythontutorial.net/python-basics/python-read-text-file/)
* [numpy.matmul — NumPy v1.24 Manual](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
* [numpy.append — NumPy v1.24 Manual](https://numpy.org/doc/stable/reference/generated/numpy.append.html#numpy.append)
* [numpy.transpose — NumPy v1.24 Manual](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
* [Indexing — NumPy v1.10 Manual](https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html#basic-slicing-and-indexing)
* [numpy.identity — NumPy v1.24 Manual](https://numpy.org/doc/stable/reference/generated/numpy.identity.html)
* [What is the meaning of "int(a\[::-1\])" in Python? - Stack Overflow](https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python)
* [\[::-1\] in Python with Examples](https://www.guru99.com/1-in-python.html)


### OpenCV:
* [rvecs](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html)
* [how can i get the camera projection matrix out of calibrateCamera() return values - Stack Overflow](https://stackoverflow.com/questions/16101747/how-can-i-get-the-camera-projection-matrix-out-of-calibratecamera-return-value)
* [Why does OpenCV think I am using a non-planar calibration rig? - Stack Overflow](https://stackoverflow.com/questions/31367255/why-does-opencv-think-i-am-using-a-non-planar-calibration-rig)
* [Camera calibration using multiple images - Signal Processing Stack Exchange](https://dsp.stackexchange.com/questions/24459/camera-calibration-using-multiple-images)
* [How to get camera focal length with OpenCV - Stack Overflow](https://stackoverflow.com/questions/58269814/how-to-get-camera-focal-length-with-opencv)
* [Camera calibration With OpenCV](https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html)
* [Perspective-n-Point (PnP) pose computation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
* [solvePnPRefineVVS() Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga17491c0282e4af874f6206a9166774a5)

### Camera Calibration:
* [Projector-Camera System Calibration and Non-planar Scene Estimation](https://uwspace.uwaterloo.ca/handle/10012/18684)
* [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
* [Camera calibration With OpenCV](https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html)
* [How to get camera focal length with OpenCV - Stack Overflow](https://stackoverflow.com/questions/58269814/how-to-get-camera-focal-length-with-opencv)

### Transformation OpenCV:
* [Perspective Transformation - Python OpenCV - GeeksforGeeks](https://www.geeksforgeeks.org/perspective-transformation-python-opencv/)
* [OpenCV: solve() Operations on arrays](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga12b43690dbd31fed96f213eefead2373)
* [CPnP:Consistent-Pose-Estimator-for-Perspective-n-Point-Problem](https://arxiv.org/pdf/2209.05824.pdf)

### Lec Notes:
* [CS491E/791E: Computer Vision (Spring 2004)](https://www.cse.unr.edu/~bebis/CS791E/Notes/CameraCalibration.pdf)
* [Computer Vision (CMU 16-385)](http://16385.courses.cs.cmu.edu/fall2022/lecture/cameras)
* [CSE/EE486 Computer Vision I](https://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf)
* [Find eigenvectors and eigenvalues of (A^T)A instead of A (differences?)](http://www.cs.cmu.edu/~16385/s15/lectures/Lecture17.pdf)
