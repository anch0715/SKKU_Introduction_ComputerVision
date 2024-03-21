import cv2
import time
import numpy as np
import A1_image_filtering as a1
from matplotlib import pyplot as plt

#가우시안 필터링
def gaussian_filtering(img, size, sigma):
    kernel = a1.get_gaussian_filter_2d(size, sigma)
    filtered_img = a1.cross_correlation_2d(img, kernel)
    return filtered_img

#Sobel 필터를 활용해 magnitude 와 direction 계산
def compute_image_gradient(img):
    Sfilterx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sfiltery = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Sxf = a1.cross_correlation_2d(img, Sfilterx)
    Syf = a1.cross_correlation_2d(img, Sfiltery)

    mag = np.sqrt(Sxf**2 + Syf**2)
    dir = np.arctan2(Syf, Sxf)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if dir[i,j] < 0:
                dir[i,j] = dir[i,j] + np.pi*2
    dir = dir*(180/np.pi)

    return mag, dir

def quanti_angle(dir):
    r, c = dir.shape
    angles = [i*45 for i in range(8)]
    ran = [(90*i+45)/2 for i in range(8)]
    for i in range(r):
        for j in range(c):
            if ran[0] <= dir[i,j] < ran[1]:
                dir[i,j] = angles[1]
            elif ran[1] <= dir[i,j] < ran[2]:
                dir[i,j] = angles[2]
            elif ran[2] <= dir[i,j] < ran[3]:
                dir[i,j] = angles[3]
            elif ran[3] <= dir[i, j] < ran[4]:
                dir[i, j] = angles[4]
            elif ran[4] <= dir[i,j] < ran[5]:
                dir[i,j] = angles[5]
            elif ran[5] <= dir[i,j] < ran[6]:
                dir[i,j] = angles[6]
            elif ran[6] <= dir[i,j] < ran[7]:
                dir[i,j] = angles[7]
            else:
                dir[i, j] = angles[0]

    return dir

def compare(dir, partial):
    angles = [i * 45 for i in range(8)]
    mag = partial[1,1]
    if (dir == angles[0]) or (dir == angles[4]):
        if mag >= (max(partial[1,0], partial[1, 2])):
            return mag
    elif (dir == angles[1]) or (dir == angles[5]):
        if mag >= (max(partial[0, 2], partial[2, 0])):
            return mag
    elif (dir == angles[2]) or (dir == angles[6]):
        if mag >= (max(partial[0, 1], partial[2, 1])):
            return mag
    elif (dir == angles[3]) or (dir == angles[7]):
        if mag >= (max(partial[0, 0], partial[2,2])):
            return mag
    return 0


def non_maximum_suppression(mag, dir):
    dir = quanti_angle(dir)
    nms_img = np.zeros(mag.shape)
    pad_mag = a1.padding(mag, (3,3))
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            partial = pad_mag[i:i+3, j:j+3]
            nms_img[i,j] = compare(dir[i,j], partial)

    return nms_img

if __name__ == "__main__":

    IMAGE_FILE_NAME = input('IMAGE FILE NAME: ')
    img = cv2.imread(IMAGE_FILE_NAME, cv2.IMREAD_GRAYSCALE)

    gfilter_img = gaussian_filtering(img, 7, 1.5)

    s_time = time.time()
    mag, dir = compute_image_gradient(gfilter_img)
    e_time = time.time()

    print("Computational time taken for computing image gradient is", e_time-s_time, 'sec')

    plt.imshow(mag, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.imwrite('./result/part_2_edge_raw_' + IMAGE_FILE_NAME, mag)


    s_time = time.time()
    nms_img = non_maximum_suppression(mag, dir)
    e_time = time.time()

    print("Computational time taken for computing non maximum suppression is", e_time - s_time, 'sec')

    plt.imshow(nms_img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.imwrite('./result/part_2_edge_sup_' + IMAGE_FILE_NAME, nms_img)