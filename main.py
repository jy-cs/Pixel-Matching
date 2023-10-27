import numpy as np
import cv2 as cv
from scipy import linalg as la
from zncc import zncc
from epipolar import calculate_matrix as epi
from epipolar import all_pixels, find_nearest_point, draw_on_left, draw_on_right


def calc_mat_m(path_2d, path_3d):
    # opening both the files in reading modes
    with open(path_2d) as f_2d, open(path_3d) as f_3d:
        num_line_2d = int(f_2d.readline().split('\n')[0])
        num_line_3d = int(f_3d.readline().split('\n')[0])
        if num_line_2d != num_line_3d:
            print("Error: Number of Points does NOT Match in 2D.text and 3D.txt")
            exit()
        # initialize matrics P and Q for input files
        P = np.ones((4, num_line_3d))
        Q = np.ones((3, num_line_3d))

        # initialize Matrix A
        A = np.zeros((num_line_3d * 2, 12))

        # fill Matrix P, Q, and A
        while num_line_2d > 0:
            # define the index of point, i
            i = num_line_3d - num_line_2d

            # define 2d coord (u,v)
            line_2d = f_2d.readline().split(' ')
            u = float(line_2d[0])
            v = float(line_2d[1])
        
            # fill matrix Q with u, v
            Q[0, i] = u
            Q[1, i] = v

            # define 3d coord (x,y,z)
            line_3d_raw = f_3d.readline().split(' ')
            line_3d_fl = []
            for j in range(len(line_3d_raw)):
                if len(line_3d_raw[j]) > 0:
                    line_3d_fl.append(float(line_3d_raw[j]))
            x = line_3d_fl[0]
            y = line_3d_fl[1]
            z = line_3d_fl[2]
        
            # fill matrix P with x, y, z
            P[0, i] = x
            P[1, i] = y
            P[2, i] = z
        
            # fill rows 2i and 2i+1 in A
            A[i * 2, 0] = -x
            A[i * 2, 1] = -y
            A[i * 2, 2] = -z
            A[i * 2, 3] = -1
            A[i * 2, 8] = u * x
            A[i * 2, 9] = u * y
            A[i * 2, 10] = u * z
            A[i * 2, 11] = u
            A[i * 2 + 1, 4] = -x
            A[i * 2 + 1, 5] = -y
            A[i * 2 + 1, 6] = -z
            A[i * 2 + 1, 7] = -1
            A[i * 2 + 1, 8] = v * x
            A[i * 2 + 1, 9] = v * y
            A[i * 2 + 1, 10] = v * z
            A[i * 2 + 1, 11] = v

            # update
            num_line_2d -= 1

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
    print("The eigenvector is valid because the error = ", la.norm(left_side - right_side),
      "is significantly small")

    # define the matrix M
    mat_m = eigenvec_s.reshape((3,4))

    # compute Q'
    Q_prime = np.matmul(mat_m, P)
    for j in range(num_line_3d):
        lamb = Q_prime[2, j]
        if lamb == 0:
            continue
        Q_prime[0, j] /= lamb
        Q_prime[1, j] /= lamb
        Q_prime[2, j] /= lamb

    # compute E_total
    n = num_line_3d
    err_total = 0
    for i in range(n):
        q_i = Q[:-1, i]
        q_i_prime = Q_prime[:-1, i]
        err_i = la.norm(q_i - q_i_prime)
        err_total += err_i

    # compute E_avg
    err_avg = err_total / n
    print("The aveage error is ", err_avg)
    return mat_m

def call_back_act(event, x, y, flags, param):
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
        pt_3d = calc_3d_reconstr(x, y, min_x, min_y)
        print("The 3D coordinates of the clicked point is ({x}, {y}, {z})".format(x=pt_3d[0],y=pt_3d[1],z=pt_3d[2]))


def calc_3d_reconstr(x, y, x_p, y_p):
    pt_reconstr = (0, 0)
    m1 = mat_m1
    m2 = mat_m2
    # Solve for A(x,y,z)=B
    a = np.array([[x*m1[2,0]-m1[0,0], x*m1[2,1]-m1[0,1], x*m1[2,2]-m1[0,2], x*m1[2,3]-m1[0,3]],
                  [y*m1[2,0]-m1[1,0], y*m1[2,1]-m1[1,1], y*m1[2,2]-m1[1,2], y*m1[2,3]-m1[1,3]],
                  [x_p*m2[2,0]-m2[0,0], x_p*m2[2,1]-m2[0,1], x_p*m2[2,2]-m2[0,2], x_p*m2[2,3]-m2[0,3]],
                  [y_p*m2[2,0]-m2[1,0], y_p*m2[2,1]-m2[1,1], y_p*m2[2,2]-m2[1,2], y_p*m2[2,3]-m2[1,3]]])
    # b = np.array([-x*m1[2,3]+m1[0,3], -y*m1[2,3]+m1[1,3], -x_p*m2[2,3]+m2[0,3], -y_p*m2[2,3]+m2[1,3]])
    # res = la.lstsq(a, b)
    # res = la.solve(np.matmul(a.transpose(),a), np.matmul(a.transpose(),b))
    m_u, m_s, m_vh = la.svd(a)
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
    left_side = np.matmul(a, eigenvec_s)
    right_side = eigenval_s * m_u[:, eigen_index]
    # print("The eigenvector is valid because the error = ", la.norm(left_side - right_side),
    #   "is significantly small")

    # define the matrix M
    # mat_m = eigenvec_s.reshape((3,4))
    # pt_reconstr = (int(res[0][0]*100)/100, int(res[0][1]*100)/100, int(res[0][2]*100)/100)
    t = eigenvec_s[3]
    pt_reconstr = (int(eigenvec_s[0]/t), int(eigenvec_s[1]/t), int(eigenvec_s[2]/t))
    return pt_reconstr

# driver function
if __name__=="__main__":
    mat_m1 = []
    mat_m2 = []
    # define paths of the two input files
    path_2d_left = "2dleft.txt"
    path_3d_left = "3dleft.txt"
    mat_m1 = calc_mat_m(path_2d_left, path_3d_left)
    path_2d_right = "2dright.txt"
    path_3d_right = "3dright.txt"
    mat_m2 = calc_mat_m(path_2d_right, path_3d_right)
    print("The camera matrix M1 is\n", mat_m1)
    print("The camera matrix M2 is\n", mat_m2)

    # read two input images as color format
    path_img1 = "im1.jpeg"
    path_img2 = "im2.jpeg"
    img1 = cv.imread(path_img1)
    img2 = cv.imread(path_img2)

    # Calculate the fundamental matrix
    pts1, pts2, F, mask = epi(img1, img2)
    print("The fundamental matrix F is\n", F)

    # Show the images
    cv.imshow('image_left', img1)
    cv.imshow('image_right', img2)

    # Set the mouse callback function
    cv.setMouseCallback('image_left', call_back_act)

    # Wait for a key to exit
    while(1):
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break


