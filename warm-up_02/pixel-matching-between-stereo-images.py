# Introduction
#   This Program Performs Pixel Matching Between a Pair of Stereo Images
#

import numpy as np
from numpy import linalg as la
import cv2 as cv


# find_fund_mat() find the fundamental matrix F given two images
def find_fund_mat(img1, img2):
    # convert images from BGR to grayscale
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # initiate SIFT detector
    sift = cv.SIFT_create()

    # detect and compute the keypoints and descriptors with SIFT
    kpt1, dest1 = sift.detectAndCompute(img1_gray, None)
    kpt2, dest2 = sift.detectAndCompute(img2_gray, None)

    # use BFMatcher.knnMatch() to get k best matches
    bf = cv.BFMatcher()
    matches = bf.knnMatch(dest1, dest2, k=2)

    # apply ratio test explained by D.Lowe in his paper
    #   adjust cutoff from 0.75 to 0.29
    good = []
    for m,n in matches:
        if m.distance < 0.29 * n.distance:
            good.append([m])
    num = len(good)
    print("\n", "Number of good matches:", num)

    # construct matrix A st. AX = 0,
    #   where X = (f_11,f_12,...,f_33)^\top
    A = np.ones((num, 9))
    for i in range(num):
        idx1 = good[i][0].queryIdx
        idx2 = good[i][0].trainIdx
        x1 = kpt1[idx1].pt[0]
        y1 = kpt1[idx1].pt[1]
        x2 = kpt2[idx2].pt[0]
        y2 = kpt2[idx2].pt[1]
        A[i][0] = x2 * x1
        A[i][1] = x2 * y1
        A[i][2] = x2
        A[i][3] = y2 * x1
        A[i][4] = y2 * y1
        A[i][5] = y2
        A[i][6] = x1
        A[i][7] = y1
    
    # apply the SVD routine s.t. A=m_u m_s m_vh
    m_u, m_s, m_vh = la.svd(A)

    # find the smallest eigenvalue
    eigen_index = 0
    eigenval_s = m_s[0]
    for i in range(m_s.shape[0]):
        if m_s[i] < eigenval_s:
            eigen_index = i
            eigenval_s = m_s[i]
    
    # find the corresponding eigenvector
    eigenvec_s = m_vh[eigen_index]

    # check the correctness based on Av=\sigma u
    left_side = np.matmul(A, eigenvec_s)
    right_side = eigenval_s * m_u[:, eigen_index]
    print("\n", "The eigenvector is valid because the error = ", la.norm(left_side - right_side), "is significantly small")

    # return the fundamental matrix F
    return eigenvec_s.reshape((3,3))

# find_epi_line() produces a lst containing the start point and
#   the end point of the epipolar line in the image given a
#    fundamental matrix and a coordinate on the other image
def find_epi_line(mat_f, pt_p):
    lst_start_end = []
    coeff = np.matmul(mat_f, pt_p)
    if coeff[0] == 0:
        y_val = int(- coeff[2] / coeff[1])
        lst_start_end.append((0, y_val))
        lst_start_end.append((width, y_val))
    elif coeff[1] == 0:
        x_val = int(- coeff[2] / coeff[0])
        lst_start_end.append((x_val, 0))
        lst_start_end.append((x_val, height))
    else:
        y_val_1 = int(- coeff[2] / coeff[1])
        pt_1 = (0, y_val_1)
        lst_start_end.append(pt_1)
        y_val_2 = int(- (coeff[2] + coeff[0] * width) / coeff[1])
        pt_2 = (width, y_val_2)
        lst_start_end.append(pt_2)
    return lst_start_end

# close_prog() terminates the program
def close_prog(event, x, y, flags, params):
    # checking for right mouse clicks
    if event == cv.EVENT_RBUTTONDOWN:
        # close the window
        cv.destroyAllWindows()
        # terminates
        exit()

# find_p_prime() produces the coordinate of p' in pixels
def find_p_prime(x, y):
    # crop the image to get a patch for mathcing
    img_source = img2_cmp
    img_template = img1_cmp
    tmpl_size = 100
    cr_x0 = x - tmpl_size
    cr_x1 = x + tmpl_size
    cr_y0 = y - tmpl_size
    cr_y1 = y + tmpl_size
    if cr_x0 < 0:
        cr_x1 -= cr_x0
        cr_x0 = 0
    elif cr_x1 > width:
        cr_x0 -= cr_x1 - width
        cr_x1 = width
    if cr_y0 < 0:
        cr_y1 -= cr_y0
        cr_y0 = 0
    elif cr_y1 > height:
        cr_y0 -= cr_y1 - height
        cr_y1 = height
    img_template = img_template[cr_y0:cr_y1, cr_x0:cr_x1]
    cv.imshow('test-img-tmpl', img_template)

    # match using matchTemplate()
    match_method = cv.TM_SQDIFF
    result = cv.matchTemplate(img_source, img_template, match_method)
    minLocation = cv.minMaxLoc(result, None)[2]
    return (minLocation[0] + tmpl_size, minLocation[1] + tmpl_size)

# draw_plus() displays a "+" at the points clicked on the image 
def draw_plus(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # '+' sign coordinates
        vpt0 = (x, y + 10)
        vpt1 = (x, y - 10)
        hpt0 = (x - 10, y)
        hpt1 = (x + 10, y)
        
        # color in BGR
        color = (69, 129, 247)
        
        # Line thickness of 2 px
        thickness = 2

        # draw "+" sign
        cv.line(img1, vpt0, vpt1, color, thickness, cv.LINE_AA)
        cv.line(img1, hpt0, hpt1, color, thickness, cv.LINE_AA)

        # draw the corresponding epipolar line
        thickness = 1
        epi_line = find_epi_line(mat_f, np.array([x, y, 1]))
        cv.line(img2, epi_line[0], epi_line[1], color, thickness, cv.LINE_AA)

        # draw the corresponding '+' on pixel p'
        p_prime = find_p_prime(x, y)
        coeff = np.matmul(mat_f, np.array([x, y, 1]))
        p_prime = (p_prime[0], (- coeff[0] * p_prime[0] - coeff[2]) / coeff[1])
        
        # '+' sign coordinates
        x = int(p_prime[0])
        y = int(p_prime[1])
        vpt0 = (x, y + 10)
        vpt1 = (x, y - 10)
        hpt0 = (x - 10, y)
        hpt1 = (x + 10, y)
        color = (110, 238, 245)
        thickness = 2
        cv.line(img2, vpt0, vpt1, color, thickness, cv.LINE_AA)
        cv.line(img2, hpt0, hpt1, color, thickness, cv.LINE_AA)

        # draw epipolar line
        cv.imshow('img1 - left', img1)
        cv.imshow('img2 - right', img2)

# driver function
if __name__=="__main__":
    ## 1 Read Images and Find F Matrix
    # define two input files' paths
    path1 = "/Users/jyang/Desktop/img_pair_08/02_7.png"
    path2 = "/Users/jyang/Desktop/img_pair_08/02_8.png"

    # read two input images as color format
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)

    # define the identical images for computation instead of drawing and showing
    img1_cmp = img1.copy()
    img2_cmp = img2.copy()

    # height, width, number of channels in image
    height = img2.shape[0]
    width = img2.shape[1]
    print("\n", width, height)

    # compute the fundamental matrix F
    mat_f = find_fund_mat(img1, img2)
    print("\n", "The fundamental matrix F is\n", mat_f)

    ## 2 Create Two Image Windows and Draw Epipolar Lines
    # displaying the images
    cv.imshow('img1 - left', img1)
    cv.imshow('img2 - right', img2)

    # setting mouse handler for the image and calling the draw_plus() function
    # cv.namedWindow('image')
    cv.setMouseCallback('img1 - left', draw_plus)
    cv.setMouseCallback('img2 - right', close_prog)

    # wait for a key to be pressed to exit
    while True:
        cv.waitKey(3 * 1000)


