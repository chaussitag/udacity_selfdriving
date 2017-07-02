#!/usr/bin/env python
# coding=utf8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

if __name__ == '__main__':
    # read image and convert to gray image
    img = mpimg.imread("exit-ramp.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    # Note: cv2.Canny actually do a gaussian blur, we just demonstrate how to use it here
    kernel_size = 5
    blurred_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    # canny edge detection
    low_threshold = 80
    high_threshold = 180
    edge_img = cv2.Canny(blurred_img, low_threshold, high_threshold)

    fig = plt.gcf()
    axs1 = fig.add_subplot(3, 1, 1, title = "gray image")
    axs1.imshow(gray_img, cmap = "gray")

    axs2 = fig.add_subplot(3, 1, 2, title = "gaussian blurred with kernel size %d" % kernel_size)
    axs2.imshow(blurred_img, cmap = "gray")

    axs3 = fig.add_subplot(3, 1, 3, title = "canny edge")
    axs3.imshow(edge_img, cmap = "Greys_r")

    plt.draw()
    plt.show()