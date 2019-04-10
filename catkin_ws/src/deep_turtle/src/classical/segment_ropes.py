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

        z = np.polyfit(ydata, xdata, 3)
        f = np.poly1d(z)

        return f

    def draw_poly(self, f, img):
        y_fit = np.arange(0, img.shape[0], 1)
        x_fit = f(y_fit)

        x_fit = map(int, x_fit)

        ymin = 0
        ymax = img.shape[1]
        xmin = 0
        xmax = img.shape[0]

        for i in range(len(x_fit)):
            yidx = y_fit[i]
            xidx = x_fit[i]

            if (xidx > xmin) and (xidx < xmax):
                # print(xidx)
                # print(yidx)
                # print(img[xidx, yidx])
                img[yidx, xidx] = [0, 0, 255]

        pt = self.get_middle_point(f, 160)
        # y_middle = img.shape[0] / 3
        # x_middle = f(y_middle)
        # x_middle = int(x_middle)
        # cv2.circle(img, (x_middle, y_middle), 10, (0, 255, 0), -1)
        cv2.circle(img, pt, 10, (0, 255, 0), -1)

    def get_middle_point(self, f, y_pt):
        y_middle = y_pt
        x_middle = f(y_middle)
        x_middle = int(x_middle)

        return (x_middle, y_middle)

    def segment_img(self, img):

        # Hue Lightness Saturation
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        margin = 25
        lower_white = np.array([0, 255 - margin, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(img_hls, lower_white, upper_white)

        i, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        mask = np.zeros_like(mask)
        for idx in range(len(cont_sorted) - 1, -1, -1):
            cv2.drawContours(mask, cont_sorted, idx, 255, -1)

        return mask / 255.


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
