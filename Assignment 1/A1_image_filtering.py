import numpy as np
import cv2
import time
from matplotlib import pyplot as plt



#1-1 Image Filtering by cross_correlation
def padding(img, shape):
    if shape[0] != 1:
        size = shape[0] // 2
        for i in range(-1, 1):
            copy = img[i]
            for j in range(size):
                img = np.insert(img, i, copy, axis=0)

    if (shape[0] == 1) or (shape[0]==shape[1]):
        size = shape[1] // 2
        for i in range(-1, 1):
            copy = img[:,i]
            for j in range(size):
                img = np.insert(img, i, copy, axis=1)

    return img


def cross_correlation_1d(img, kernel):
    filter_img = np.zeros(img.shape)
    pad_img = padding(img, np.shape(kernel))
    r, c = img.shape
    # horizontal case
    if np.shape(kernel)[0] == 1:
        size = kernel.shape[1]
        for i in range(0, r):
            for j in range(0, c):
                partial = np.reshape(pad_img[i, j:j + size], kernel.shape)
                filter_img[i, j] = (kernel * partial).sum()

    # vertical case
    else:
        size = kernel.shape[0]
        for i in range(0, r):
            for j in range(0, c):
                partial = np.reshape(pad_img[i:i + size, j], kernel.shape)
                filter_img[i, j] = (kernel * partial).sum()

    return filter_img


def cross_correlation_2d(img, kernel):
    size = len(kernel)
    filter_img = np.zeros(img.shape)
    pad_img = padding(img, np.shape(kernel))
    r, c = img.shape
    for i in range(0, r):
      for j in range(0, c):
        partial = pad_img[i:i+size, j:j+size]
        filter_img[i,j] = (kernel * partial).sum()

    return filter_img

#1-2 The Gaussian Filter

def get_gaussian_filter_1d(size, sigma):
    kernel = np.zeros(size)
    for i in range(size):
        kernel[i] = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-1*((i-(size//2))**2)/(2*(sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel.reshape(size,1)

def get_gaussian_filter_2d(size, sigma):
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*(sigma**2)))*np.exp(-1*((((i-(size//2))**2)+((j-(size//2))**2))/(2*(sigma**2))))
    kernel = kernel/kernel.sum()
    return kernel



if __name__ == "__main__":

    kernel1d = get_gaussian_filter_1d(5, 1)
    kernel2d = get_gaussian_filter_2d(5, 1)

    #가우시안 kernel 출력
    print('1D Kernel and 2D Kernel size 5, sigma 1 ')
    print(kernel1d)
    print(kernel2d)


    #kernel size 와 sigma 를 다르게 해서 필터된 이미지 출력
    kernel_size = [5,11,17]
    kernel_sigma = [1, 6, 11]

    IMAGE_FILE_NAME = input('IMAGE FILE NAME: ')

    img = cv2.imread(IMAGE_FILE_NAME, cv2.IMREAD_GRAYSCALE)

    images = []

    #kernel size와 sigma에 따른 필터 적용 후 text caption 추가

    for i in kernel_size:
        for j in kernel_sigma:
            kernel = get_gaussian_filter_2d(i, j)
            filter_img = cross_correlation_2d(img, kernel)
            text = str(i)+'x'+str(i)+' s='+ str(j)
            filter_img = cv2.putText(filter_img, text, (10,40), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale = 1, color = (0, 0, 0),
                                     thickness = 2)

            images.append(filter_img)

    #kernel size 에 따라 아미지 이어서 붙이기

    for i in range(3):
        images[i] = np.concatenate((images[i*3], images[i*3+1], images[i*3+2]), axis=1)

    #kernel sigma 에 따라 아미지 이어서 붙이기
    filter_img = np.concatenate((images[0], images[1], images[2]), axis=0)

    #이미지 출력
    plt.imshow(filter_img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    #이미지 저장
    cv2.imwrite('./result/part_1_gaussian_filtered_'+IMAGE_FILE_NAME, filter_img)


    #1차원 kernel(17, 6) 수직,수평 수행 및 시간 기록
    kernel1d = get_gaussian_filter_1d(5, 6)

    start_time = time.time()
    filtered_by_1d = cross_correlation_1d(img, kernel1d)
    filtered_by_1d = cross_correlation_1d(filtered_by_1d, kernel1d.T)
    end_time = time.time()
    time_1d = end_time - start_time


    #2차원 kernel(5, 6) 수행 및 시간 기록
    kernel2d = get_gaussian_filter_2d(5, 6)

    start_time = time.time()
    filtered_by_2d = cross_correlation_2d(img, kernel2d)
    end_time = time.time()
    time_2d = end_time - start_time



    pixel_wise_map = cv2.absdiff(filtered_by_1d,filtered_by_2d)

    plt.imshow(pixel_wise_map, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


    print('The sum of absolute intensity differences: ',cv2.absdiff(filtered_by_1d,filtered_by_2d).sum())


    print('Computational time of Gaussian filtering by applying vertical and horizontal 1D kernel: ', time_1d, 'sec')
    print('Computational time of Gaussian filtering by applying 2D kernel: ', time_2d, 'sec')

