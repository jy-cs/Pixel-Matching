############################################
# COMP8510 Project-02
# Group Information:
#   Jiajie Yang [yang4q, 110115897]
#   Jilsa Chandarana [chandarj, 110105879]
############################################
#
# Introduction
#   This Program Performs Pixel Matching Between a Pair of Stereo Images
#
# Algorithm
# 1. Find the feature points using SIFT
# 2. Calculate the fundamental matrix
# 3. Calculate the epipolar line of selected point
# 4. Find the nearest feature point and calculate the relative distance for selected point
# 5. The pixels that fall on the epipolar line and close to the relative distance of the selected point are considered and ZNCC score is calculated for them
# 6. Pixel with the lowest ZNCC score is selected and marked on the right image

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.draw import line
from zncc import zncc

def calculate_matrix(img1, img2):
    # Find feature points using SIFT
    sift = cv.SIFT_create()

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_gray,None)
    kp2, des2 = sift.detectAndCompute(img2_gray,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 100)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    # Taking Low ratio to improve accuracy
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.4*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Find the Fundamental Matrix
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    return pts1, pts2, F, mask


def find_nearest_point(x, y, pts1, pts2):
    # Return the nearest points
    dist = np.sqrt((pts1[:, 0] - x)**2 + (pts1[:, 1] - y)**2)
    idx = np.argmin(dist)
    return pts1[idx, :], pts2[idx, :]


def all_pixels(img, line_co):
    # For a given line coefficients a, b, c of ax+by+c = 0, find all pixels that lie on the line
    x1 = x2 = y1 = y2 = -1

    final_x = []
    final_y = []

    # For (x, 1)
    x = -(line_co[2] + line_co[1] * 1) / line_co[0]
    if(x > 0 and x < img.shape[1]):
        final_x.append(int(x))
        final_y.append(1)

    # For (1, y)
    y = -(line_co[2] + line_co[0] * 1) / line_co[1]
    if(y > 0 and y < img.shape[0]):
        final_x.append(1)
        final_y.append(int(y))

    # For (x, max_y)
    x = -(line_co[2] + line_co[1] * img.shape[0]) / line_co[0]
    if(x > 0 and x < img.shape[1]):
        final_x.append(int(x))
        final_y.append(img.shape[0])

    # For (max_x, y)
    y = -(line_co[2] + line_co[0] * img.shape[1]) / line_co[1]
    if(y > 0 and y < img.shape[0]):
        final_x.append(img.shape[1])
        final_y.append(int(y))

    rr, cc = line(final_x[0], final_y[0], final_x[-1], final_y[-1])
    return list(zip(rr, cc))


def draw_on_left(img, x, y, color):
    # Draw a plus sign on the left image
    img1 = img.copy()
    img1 = cv.line(img1, (x-5, y), (x+5, y), color, 2)
    img1 = cv.line(img1, (x, y-5), (x, y+5), color, 2)

    cv.imshow('image_left', img1)


def draw_on_right(img, x, y, epi_line, epi_x, epi_y, color):
    img2 = img.copy()
    a = epi_line[0][0]
    b = epi_line[0][1]
    c = epi_line[0][2]
    width = img2.shape[1]
    epi_start = [0, int(-c/b)]
    epi_end = [width, int((-c-a*width)/b)]
    # Draw a epipolar on the right image
    img2 = cv.line(img2, epi_start, epi_end, color, 1)
    y = int((-c-a*x)/b)
    # Draw a plus sign on the right image
    img2 = cv.line(img2, (x-5, y), (x+5, y), color, 2)
    img2 = cv.line(img2, (x, y-5), (x, y+5), color, 2)
    cv.imshow('image_right', img2)


def final_call(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Calculate the Epipolar Line
        epi_line = cv.computeCorrespondEpilines(np.array([(x, y)], dtype=np.float32), 1, F)
        epi_line = epi_line.reshape(-1, 3)

        # Find list of all pixels that lie on the epipolar line
        list_of_points = all_pixels(img1, epi_line[0, :])

        # Find the closest feature point
        closest_pts1, closest_pts2 = find_nearest_point(x, y, pts1, pts2)

        # Calculate the relative distance between the feature point and selected point
        diff_x, diff_y = closest_pts1[0] - x, closest_pts1[1] - y
        difference_window = 10

        # Get all the pixels that fall within selected range
        x_range = np.arange(diff_x - difference_window, diff_x + difference_window + 1)
        y_range = np.arange(diff_y - difference_window, diff_y + difference_window + 1)

        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


        # Calculate the ZNCC value for the points in the selected range
        zncc_window_size = 3
        min_zncc = 99999
        min_x, min_y = -1, -1
        for i in range(len(list_of_points)):
            if(int(closest_pts2[0] - list_of_points[i][0]) in x_range and int(closest_pts2[1] - list_of_points[i][1]) in y_range):
                zncc_value = zncc(img1_gray, img2_gray, x, y, list_of_points[i][0], list_of_points[i][1], zncc_window_size)
                if zncc_value < min_zncc:
                    min_zncc = zncc_value
                    min_x, min_y = list_of_points[i][0], list_of_points[i][1]

        # If the pixel has similar neightbors, get the point relative to the feature point
        if(min_x== -1 and min_y== -1):
            min_x = closest_pts2[0] - diff_x
            min_y = closest_pts2[1] - diff_y

        # Draw plus sign on the images
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw_on_left(img1, x, y, color)
        draw_on_right(img2, min_x, min_y, epi_line, x, y, color)




if __name__ == '__main__':
    # Load the images
    img1 = cv.imread('../img_pair_03/comp851-proj02_1.png')
    img2 = cv.imread('../img_pair_03/comp851-proj02_1.png')

    # Calculate the fundamental matrix
    pts1, pts2, F, mask = calculate_matrix(img1, img2)

    # Show the images
    cv.imshow('image_left', img1)
    cv.imshow('image_right', img2)

    # Set the mouse callback function
    cv.setMouseCallback('image_left', final_call)

    # Wait for a key to exit
    while(1):
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break