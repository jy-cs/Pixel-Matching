#!/usr/bin/env python

"""
Calculate a characteristic form images called ZNCC.
(Zero-Normalized Cross Correlation?)
"""


def get_average(img, u, v, n):
    """img as a square matrix of numbers"""
    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if(u+i > 0 and v+j > 0 and u+i < img.shape[0] and v+j < img.shape[1]):
                s += img[u+i][v+j]

    return float(s)/(2*n+1)**2


def get_standard_deviation(img, u, v, n):
    """
    Get the standard deviation of the n-pixel range around (u,v).
    Parameters
    ----------
    img
    u : int
    v : int
    n : int
    Returns
    -------
    float
    """
    s = 0
    avg = get_average(img, u, v, n)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if(u+i > 0 and v+j > 0 and u+i < img.shape[0] and v+j < img.shape[1]):
                s += (img[u+i][v+j] - avg)**2

    return (s**0.5)/(2*n+1)


def zncc(img1, img2, u1, v1, u2, v2, n):
    """
    Calculate the ZNCC value for img1 and img2.
    """
    std_deviation1 = get_standard_deviation(img1, v1, u1, n)
    std_deviation2 = get_standard_deviation(img2, v2, u2, n)
    avg1 = get_average(img1, v1, u1, n)
    avg2 = get_average(img2, v2, u2, n)


    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if(u1+i > 0 and v1+j > 0 and u1+i < img1.shape[0] and v1+j < img1.shape[1] and u2+i > 0 and v2+j > 0 and u2+i < img2.shape[0] and v2+j < img2.shape[1]):
                s += (img1[u1+i][v1+j] - avg1)*(img2[u2+i][v2+j] - avg2)

    if(std_deviation1 == 0 or std_deviation2 == 0):
        return 99999
    return float(s)/((2*n+1)**2 * std_deviation1 * std_deviation2)
