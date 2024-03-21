import cv2
import time
import numpy as np
import A1_image_filtering as a1
from matplotlib import pyplot as plt

def gaussian_filtering(img, size, sigma):
    kernel = a1.get_gaussian_filter_2d(size, sigma)
    filtered_img = a1.cross_correlation_2d(img, kernel)
    return filtered_img


def compute_corner_response(img):
    Sfilter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sfilter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Sxf = a1.cross_correlation_2d(img, Sfilter_x)
    Syf = a1.cross_correlation_2d(img, Sfilter_y)

    Sxf = a1.padding(Sxf, (5, 5))
    Syf = a1.padding(Syf, (5, 5))

    Ix_2 = Sxf ** 2
    Iy_2 = Syf ** 2
    Ix_Iy = Sxf * Syf

    R = np.zeros(img.shape)

    w_size = 5
    r, c = img.shape
    for i in range(0, r):
        for j in range(0, c):
            M = np.zeros((2, 2))
            w_Ix_2 = Ix_2[i:i + w_size, j:j + w_size]
            w_Iy_2 = Iy_2[i:i + w_size, j:j + w_size]
            w_Ix_Iy = Ix_Iy[i:i + w_size, j:j + w_size]

            M[0, 0] = w_Ix_2.sum()
            M[1, 1] = w_Iy_2.sum()
            M[0, 1] = w_Ix_Iy.sum()
            M[1, 0] = w_Ix_Iy.sum()

            M_det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            M_trace = M[0, 0] + M[1, 1]

            R[i, j] = M_det - 0.04 * (M_trace ** 2)

    R = np.where(R < 0, 0, R)
    R = R / R.max()

    return R

def non_maximum_suppression_win(R, winSize):
    pad_R = a1.padding(R, (winSize, winSize))
    suppressed = np.zeros(R.shape)
    r, c = R.shape
    for i in range(r):
        for j in range(c):
            maximum = pad_R[i:i + winSize, j:j + winSize].max()
            if (R[i, j] == maximum) and (R[i, j] >= 0.1):
                suppressed[i, j] = R[i, j]

    return suppressed



if __name__ == "__main__":
    IMAGE_FILE_NAME = input('IMAGE FILE NAME: ')
    img = cv2.imread(IMAGE_FILE_NAME, cv2.IMREAD_GRAYSCALE)

    #코너 검출기
    gfilter_img = gaussian_filtering(img, 7, 1.5)

    s_time = time.time()
    R = compute_corner_response(gfilter_img)
    e_time = time.time()

    print('Computational time of compute_corner_response is', e_time - s_time, 'sec')

    plt.imshow(R, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    R_img = cv2.convertScaleAbs(R, alpha=(255.0))

    cv2.imwrite('./result/part_3_corner_raw_' + IMAGE_FILE_NAME, R_img)

    #검출한 코너를 초록색으로 표시
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_rgb[R > 0.1] = [0, 255, 0]

    plt.imshow(img_rgb, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.imwrite('./result/part_3_corner_bin_' + IMAGE_FILE_NAME, img_rgb)

    #Non Maximum Suppression
    winSize = 11

    s_time = time.time()
    suppressed_R = non_maximum_suppression_win(R, winSize)
    e_time = time.time()

    print('Computational time of non_maximum_suppression_win is', e_time - s_time, 'sec')


    idx_row = np.where(suppressed_R != 0)[1]
    idx_col = np.where(suppressed_R != 0)[0]

    #원 그리기
    img_circle = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (r, c) in zip(idx_row, idx_col):
        img_circle = cv2.circle(img_circle, (r, c), 4, (0, 255, 0), 2)

    plt.imshow(img_circle, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.imwrite('./result/part_3_corner_sup_' + IMAGE_FILE_NAME, img_circle)

