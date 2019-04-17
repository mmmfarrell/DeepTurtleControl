#!/usr/bin/env python
from __future__ import print_function

import numpy as np

import os
import cv2
import json

class Segmenter():

    def __init__(self):
        pass


    def get_poly_from_mask(self, mask):
        ydata, xdata = np.where(mask == 255)

        z = np.polyfit(ydata, xdata, 2)
        f = np.poly1d(z)

        return f

    def draw_poly(self, f, img):
        y_fit = np.arange(0, 290, 1)
        x_fit = f(y_fit)

        x_fit = map(int, x_fit)

        ymin = 0
        ymax = img.shape[1]
        xmin = 0
        xmax = img.shape[0]

        for i in range(len(x_fit)):
            cv2.circle(img, (x_fit[i], y_fit[i]), 2, (0, 0, 255), -1)

        pt = self.get_middle_point(f, 160)
        cv2.circle(img, pt, 10, (0, 255, 0), -1)

    def get_middle_point(self, f, y_pt):
        y_middle = y_pt
        x_middle = f(y_middle)
        x_middle = int(x_middle)

        return (x_middle, y_middle)

    def segment_img(self, img):

        img = cv2.rectangle(img, (0, 290), (640, 480), (0, 0, 0), -1)
        img = cv2.rectangle(img, (0, 0), (640, 100), (0, 0, 0), -1)

        # Hue Lightness Saturation
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        margin = 1
        lower_white = np.array([0, 255 - margin, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(img_hls, lower_white, upper_white)
        # cv2.imshow("mask", mask)

        i, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        y_pt = img.shape[0] / 3
        #y_pt = img.shape[0] / 2
        middles = {}
        buff = 0
        middles['left'] = (-buff, y_pt)
        middles['right'] = (640+buff, y_pt)
        # left_idxs = [200:480, 0:250]
        for idx in range(len(cont_sorted) - 1, -1, -1):
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, cont_sorted, idx, 255, -1)
            left_lane = np.any(mask[100:290, 0:240])
            right_lane = np.any(mask[100:290, 240:640])

            if left_lane and right_lane:
                left_idxs = np.where(np.any(mask[100:290, 0:240]))[0]
                right_idxs = np.where(np.any(mask[100:290, 240:640]))[0]

                print("ERROR, both right and left")
                if left_idxs.max() > right_idxs.max():
                    right_lane = False
                    print("I think its the right lane though")
                else:
                    left_lane = False
                    print("I think its the left lane though")

            # if left_lane:
                # cv2.imshow("left", mask)
            # if right_lane:
                # cv2.imshow("right", mask)

            # print("Contour area: ", cv2.contourArea(cont_sorted[idx]))
            cont_area = cv2.contourArea(cont_sorted[idx])
            # cont_thresh = 2000.
            cont_thresh = 500.
            if cont_area > cont_thresh:
                if right_lane:
                    cv2.circle(img, (490, 385), 20, (255, 0, 0), -1)
                if left_lane:
                    cv2.circle(img, (75, 385), 20, (255, 0, 0), -1)
                poly = self.get_poly_from_mask(mask)
                self.draw_poly(poly, img)

                if left_lane:
                    middles['left'] = self.get_middle_point(poly, y_pt)
                if right_lane:
                    middles['right'] = self.get_middle_point(poly, y_pt)

        avg_middle_x = (middles['left'][0] + middles['right'][0]) / 2
        avg_middle_y = (middles['left'][1] + middles['right'][1]) / 2
        cv2.circle(img, (avg_middle_x, avg_middle_y), 10, (0, 0, 255), -1)

        # Draw left and right lane detection
        img = cv2.rectangle(img, (0, 100), (240, 290), (0, 0, 255), 3)
        img = cv2.rectangle(img, (640, 100), (240, 290), (0, 0, 255), 3)

        return img, avg_middle_x


if __name__ == '__main__':
    seg = Segmenter()

    img_files = ["./example.jpg", "./example_hard.jpg"]
    # img_files = ["./example.jpg"]
    for img_file in img_files:
        img = cv2.imread(img_file)
        seg_img = seg.segment_img(img)

        cv2.imshow("img", img)
        cv2.imshow("segmented", seg_img)
        cv2.waitKey(0)
