#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


if __name__ == '__main__':
    img = mpimg.imread('test.jpg')
    print("This image is: ", type(img), 'with dimensions: ', img.shape)
    h = img.shape[0]
    w = img.shape[1]
    color_select = np.copy(img)
    rgb_threshold = [210, 220, 210]
    filter_out_indices = (img[:, :, 0] < rgb_threshold[0]) | (img[:, :, 1] < rgb_threshold[1]) | (img[:, :, 2] < rgb_threshold[2])
    print(filter_out_indices.shape)
    color_select[filter_out_indices] = 0

    fig = plt.gcf()
    # draw the R channel
    r_channel = np.zeros(img.shape, dtype = img.dtype)
    r_channel[:, :, 0] = img[:, :, 0]
    axs_r = fig.add_subplot(2, 3, 1, title = "R Channel")
    axs_r.imshow(r_channel)

    # draw the G channel
    g_channel = np.zeros(img.shape, dtype = img.dtype)
    g_channel[:, :, 1] = img[:, :, 1]
    axs_g = fig.add_subplot(2, 3, 2, title = "G Channel")
    axs_g.imshow(g_channel)

    # draw the B channle
    b_channel = np.zeros(img.shape, dtype = img.dtype)
    b_channel[:, :, 2] = img[:, :, 2]
    axs_b = fig.add_subplot(2, 3, 3, title = "B Channel")
    axs_b.imshow(b_channel)

    # draw orignal image
    axs_img = fig.add_subplot(2, 2, 3, title = "original image")
    axs_img.imshow(img)

    # draw the filtered image
    axs_filtered_img = fig.add_subplot(2, 2, 4, title = "filtered image")
    axs_filtered_img.text(965, 50, "threshold_r = %d" % rgb_threshold[0])
    axs_filtered_img.text(965, 100, "threshold_g = %d" % rgb_threshold[1])
    axs_filtered_img.text(965, 150, "threshold_b = %d" % rgb_threshold[2])
    axs_filtered_img.imshow(color_select)

    plt.draw()
    plt.show()
