#!/usr/bin/env python
#coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

if __name__ == '__main__':
    img = mpimg.imread("test.jpg")
    h = img.shape[0]
    w = img.shape[1]

    color_select = np.copy(img)
    rgb_threshold = [210, 220, 210]
    filter_out_indices = (img[:, :, 0] < rgb_threshold[0]) | (img[:, :, 1] < rgb_threshold[1]) | (img[:, :, 2] < rgb_threshold[2])
    print(filter_out_indices.shape)
    color_select[filter_out_indices] = 0

    left_bottom = [0, h - 1]
    right_bottom = [900, 300]
    apex = [400, 0]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    region_thresholds = (yy > (fit_left[0] * xx + fit_left[1])) & \
                        (yy > (fit_right[0] * xx + fit_right[1])) & \
                        (yy < (fit_bottom[0] * xx + fit_bottom[1]))

    region_select = np.copy(img)
    region_select[region_thresholds] = [255, 0, 0]

    fig = plt.gcf()
    ax1 = fig.add_subplot(1, 2, 1, title = "original image")
    ax1.imshow(img)

    ax2 = fig.add_subplot(1, 2, 2, title = "interested regions marked as red")
    ax2.imshow(region_select)

    plt.draw()
    plt.show()
