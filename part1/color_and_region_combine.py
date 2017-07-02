#!/usr/bin/env python
#coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

if __name__ == '__main__':
    img = mpimg.imread("test.jpg")
    h = img.shape[0]
    w = img.shape[1]

    # filter by color
    color_select = np.copy(img)
    rgb_threshold = [200, 200, 200]
    color_filter_out_indices = (img[:, :, 0] < rgb_threshold[0]) | \
                               (img[:, :, 1] < rgb_threshold[1]) | \
                               (img[:, :, 2] < rgb_threshold[2])
    print(color_filter_out_indices.shape)
    color_select[color_filter_out_indices] = 0

    # filter by a triangle region mask
    left_bottom = [100, h - 1]
    right_bottom = [810, h - 1]
    apex = [470, 310]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    interested_region_indices = (yy > (fit_left[0] * xx + fit_left[1])) & \
                                (yy > (fit_right[0] * xx + fit_right[1])) & \
                                (yy < (fit_bottom[0] * xx + fit_bottom[1]))

    region_select = np.copy(img)
    region_select[~interested_region_indices] = [0, 0, 0]

    # combine color and region mask to filter out the lane line area
    color_region_combined_select = np.copy(img)
    color_region_combined_select[~color_filter_out_indices & interested_region_indices] = [255, 0, 0]

    color_region_combined_select1 = np.copy(img)
    color_region_combined_select1[color_filter_out_indices | ~interested_region_indices] = 0

    fig = plt.gcf()
    ax1 = fig.add_subplot(3, 1, 1, title = "original image")
    ax1.imshow(img)

    ax3 = fig.add_subplot(3, 2, 3, title = "color filtered image")
    ax3.imshow(color_select)

    ax4 = fig.add_subplot(3, 2, 4, title = "interested region")
    ax4.imshow(region_select)

    ax5 = fig.add_subplot(3, 2, 5, title = "combined(show as red)")
    ax5.imshow(color_region_combined_select)

    ax6 = fig.add_subplot(3, 2, 6, title = "combined(only keep interested region)")
    ax6.imshow(color_region_combined_select1)

    plt.draw()
    plt.show()
