#!/usr/bin/env python
#coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

if __name__ == '__main__':
    img = mpimg.imread("test.jpg")
    h = img.shape[0]
    w = img.shape[1]

    # vertices of the interested triangle region
    left_bottom = [0, h - 1]
    right_bottom = [900, 300]
    apex = [400, 0]

    # line segments of the interested triangle region
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    interested_region_indices = (yy > (fit_left[0] * xx + fit_left[1])) & \
                                (yy > (fit_right[0] * xx + fit_right[1])) & \
                                (yy < (fit_bottom[0] * xx + fit_bottom[1]))

    region_select = np.copy(img)
    region_select[~interested_region_indices] = [0, 0, 0]

    fig = plt.gcf()
    ax1 = fig.add_subplot(1, 2, 1, title = "original image")
    ax1.imshow(img)

    ax2 = fig.add_subplot(1, 2, 2, title = "interested region")
    ax2.imshow(region_select)

    plt.draw()
    plt.show()
