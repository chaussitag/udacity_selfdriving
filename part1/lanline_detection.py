#!/usr/bin/env python
# coding=utf8

import cv2
import numpy as np

import argparse
import os
import sys

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def merge_line_segments(lines, h, w):
    ####################################################################################
    # separate detected lines into two groups according to position and slope of a line
    # the x coordinate that distinct left and right regions
    mid_x = int(0.50 * w)
    # each line was represented as [x1, y1, x2, y2, slope, square_len]
    left_lines = np.zeros((0, 6), dtype=lines.dtype)
    right_lines = np.zeros((0, 6), dtype=lines.dtype)
    # minimum and maximum allowed angles a line segment relative to x-aixs
    min_angle = (25.0 / 180.0) * np.pi
    max_angle = (55.0 / 180.0) * np.pi

    for line in lines:
        for x1, y1, x2, y2 in line:
            if np.abs(x1 - x2) < 3:
                continue
            slope = (y2 - y1 + 0.0) / (x2 - x1 + 0.0)
            square_len = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

            # left lines should satify following two conditions:
            #      stay to the left of (mid_x - 0.04*w) and,
            #      has a slope between [-np.tan(max_angle), -np.tan(min_angle)]
            # right lines should satify following two conditions:
            #      stay to the right of (mid_x + 0.04*4) and,
            #      has as slope between [np.tan(min_angle), np.tan(max_angle)]
            if max(x1, x2) < (mid_x - 0.04 * w) \
                    and -np.tan(max_angle) <= slope <= -np.tan(min_angle):
                left_lines = np.vstack((left_lines, [x1, y1, x2, y2, slope, square_len]))
            elif min(x1, x2) > (mid_x + 0.04 * w) \
                    and np.tan(min_angle) <= slope <= np.tan(max_angle):
                right_lines = np.vstack((right_lines, [x1, y1, x2, y2, slope, square_len]))
    ####################################################################################

    ####################################################################################
    # find the left line
    # average slopes of left line segments as the result slope for the left lane line
    left_slope_avg = np.average(left_lines[:, 4])
    # sort by line length
    left_sort_by_len_indices = np.argsort(left_lines[:, 5])

    # find the point the left line cross through
    left_lines_pt_x = 0.0
    left_lines_pt_y = 0.0
    # select top_n longest line segments, and avarage the vertices of these line segments,
    # use the averaged vertex as a point that the left lane pass through
    top_n = 3 if left_sort_by_len_indices.shape[0] > 3 else left_sort_by_len_indices.shape[0]
    for i in left_sort_by_len_indices[-top_n:]:
        left_lines_pt_x += (left_lines[i][0] + left_lines[i][2]) / 2.0
        left_lines_pt_y += (left_lines[i][1] + left_lines[i][3]) / 2.0
    left_lines_pt_x /= top_n
    left_lines_pt_y /= top_n
    assert (0 < left_lines_pt_x < w)
    assert (0 < left_lines_pt_y < h)
    # from the line equation y = slope * x + b, we get b = y - slope * x
    # 'fit_left' has form of [slope, b]
    fit_left = np.array([left_slope_avg, left_lines_pt_y - left_slope_avg * left_lines_pt_x])
    ####################################################################################

    ####################################################################################
    # find the right line
    # average slopes of left line segments as the result slope for the right lane line
    right_slope_avg = np.average(right_lines[:, 4])
    right_sort_by_len_indices = np.argsort(right_lines[:, 5])
    # find the point the left line cross through
    right_lines_pt_x = 0.0
    right_lines_pt_y = 0.0
    top_n = 3 if right_sort_by_len_indices.shape[0] > 3 else right_sort_by_len_indices.shape[0]
    for i in right_sort_by_len_indices[-top_n:]:
        right_lines_pt_x += (right_lines[i][0] + right_lines[i][2]) / 2.0
        right_lines_pt_y += (right_lines[i][1] + right_lines[i][3]) / 2.0
    right_lines_pt_x /= top_n
    right_lines_pt_y /= top_n
    assert (0 < right_lines_pt_x < w)
    assert (0 < right_lines_pt_y < h)
    fit_right = np.array([right_slope_avg, right_lines_pt_y - right_slope_avg * right_lines_pt_x])
    ####################################################################################

    return fit_left, fit_right


def draw_lines(img, lines, color=[0, 255, 255], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        x1, y1, x2, y2 = line[0:4]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def create_laneline_image(line_left, line_right, h, w):
    bottom_y = h
    top_y = int(0.65 * h)
    lines = [ \
        [int((bottom_y - line_left[1]) / line_left[0]), bottom_y, \
         int((top_y - line_left[1]) / line_left[0]), top_y], \
        [int((bottom_y - line_right[1]) / line_right[0]), \
         bottom_y, int((top_y - line_right[1]) / line_right[0]), top_y], \
        ]

    line_img = np.zeros((h, w, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    h = img.shape[0]
    w = img.shape[1]
    fit_left, fit_right = merge_line_segments(lines, h, w)

    line_img = create_laneline_image(fit_left, fit_right, h, w)

    return line_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def detect_lanelines(img):
    # height and width of the image
    h = img.shape[0]
    w = img.shape[1]
    # print("h: %d, w: %d" % (h, w))

    # convert input image to grayscale
    gray_img = grayscale(img)

    # gaussian blur
    kernel_size = 5
    blurred_img = gaussian_blur(gray_img, kernel_size)

    # canny edge detection
    low_threshold = 10
    high_threshold = 100
    edge_img = canny(blurred_img, low_threshold, high_threshold)

    # define a polygon mask to filter out unrelated edges
    polygons_vertices = \
        np.array([ \
            [(int(0.08 * w), h), (int(0.435 * w), int(0.625 * h)), \
             (int(0.585 * w), int(0.625 * h)), (int(0.95 * w), h), \
             (int(0.80 * w), h), (int(0.5 * w), int(0.625 * h)), (int(0.25 * w), h)]
        ], dtype=np.int32)
    mask_filterred_img = region_of_interest(edge_img, polygons_vertices)

    # Hough transform
    rho = 2
    theta = 1 * np.pi / 180
    threshold = 6
    min_line_length = 20
    max_line_gap = 8
    lines = cv2.HoughLinesP(mask_filterred_img, rho, theta, threshold, np.array([]), \
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    h = img.shape[0]
    w = img.shape[1]
    fit_left, fit_right = merge_line_segments(lines, h, w)

    return fit_left, fit_right


def process_single_img_file(path):
    # open the image
    img = cv2.imread(path)
    return process_single_image(img)


def process_single_image(img):
    h = img.shape[0]
    w = img.shape[1]
    fit_left, fit_right = detect_lanelines(img)
    line_img = create_laneline_image(fit_left, fit_right, h, w)

    return weighted_img(line_img, img)


class LanelineDetector(object):
    def __init__(self):
        # stores previous two frames' result for smoothing
        self._prev_left_lines = np.zeros((0, 2), dtype=np.float32)
        self._prev_right_lines = np.zeros((0, 2), dtype=np.float32)
        self._factors = np.array([0.20, 0.30, 0.50])

    def process_img(self, image):
        line_left, line_right = detect_lanelines(image)

        if self._prev_left_lines.shape[0] < 2:
            self._prev_left_lines = np.vstack((self._prev_left_lines, line_left))
            self._prev_right_lines = np.vstack((self._prev_right_lines, line_right))
        else:
            # smoothing the current result with previous two frames's result
            line_left = self._prev_left_lines[0] * self._factors[0] \
                        + self._prev_left_lines[1] * self._factors[1] \
                        + line_left * self._factors[2]
            line_right = self._prev_right_lines[0] * self._factors[0] \
                         + self._prev_right_lines[1] * self._factors[1] \
                         + line_right * self._factors[2]
            self._prev_left_lines[0] = self._prev_left_lines[1]
            self._prev_left_lines[1] = line_left
            self._prev_right_lines[0] = self._prev_right_lines[1]
            self._prev_right_lines[1] = line_right

        h = image.shape[0]
        w = image.shape[1]
        line_img = create_laneline_image(line_left, line_right, h, w)
        return weighted_img(line_img, image)


def get_process_image_func():
    detector = LanelineDetector()

    def process_image_impl(image):
        return detector.process_img(image)

    return process_image_impl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="lane line detector")
    parser.add_argument("--input_video", "-i", dest="input_video", help="path to the input video")
    parser.add_argument("--output_video", "-o", dest="output_video", help="path to the output result video")
    args = parser.parse_args()

    if len(sys.argv) == 1 or not os.path.exists(args.input_video):
        print("the input video %s does not exist" % (args.input_video,))
        parser.print_help()
        sys.exit(-1)

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(args.input_video)
    process_image_func = get_process_image_func()
    white_clip = clip1.fl_image(process_image_func)  # NOTE: this function expects color images!!
    white_clip.write_videofile(args.output_video, audio=False)