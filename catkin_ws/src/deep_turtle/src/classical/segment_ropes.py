#!/usr/bin/env python
from __future__ import print_function

import numpy as np

import os
import cv2
import json

class Segmenter():

    def __init__(self):
        pass

    def get_angle_from_contour_mask(self, mask):
        lines = cv2.HoughLines(mask,1,np.pi/180,200)

        if lines is None:
            return None

        avg_theta = 0.
        avg_rho = 0.
        for i in range(len(lines)):
                for rho,theta in lines[i]:
                    avg_theta += theta / len(lines)
                    avg_rho += rho / len(lines)

                    # a = np.cos(theta)
                    # b = np.sin(theta)
                    # x0 = a*rho
                    # y0 = b*rho
                    # x1 = int(x0 + 1000*(-b))
                    # y1 = int(y0 + 1000*(a))
                    # x2 = int(x0 - 1000*(-b))
                    # y2 = int(y0 - 1000*(a))

                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        theta = avg_theta
        rho = avg_rho
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        # print(avg_theta)
        return avg_theta

    def draw_avg_line(self, img, avg_angle):
        # pass
        x0 = img.shape[1] / 2
        y0 = img.shape[0]
        print(img.shape)

        a = np.cos(avg_angle)
        b = np.sin(avg_angle)
        # x1 = int(x0 + 1000*(-b))
        # y1 = int(y0 + 1000*(a))
        # print("xo: ", x0)
        # print("yo: ", y0)
        # print("x1: ", x1)
        # print("y1: ", y1)
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        # cv2.line(img,(x0,y0),(x1,y1),(0,0,255),2)
        return img



    def segment_img(self, img):

        # img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        margin = 15
        # lower_white = np.array([0, 0, 255 - margin], dtype=np.uint8)
        # upper_white = np.array([255, margin, 255], dtype=np.uint8)
        lower_white = np.array([0, 255 - margin, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(img_hsv, lower_white, upper_white)

        # Crop out the robot itself
        # mask[:, 10:50] = 0.


        i, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # for i in range(2):
            # x,y,w,h = cv2.boundingRect(cont_sorted[i])

            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)

        mask = np.zeros_like(mask)
        cv2.drawContours(mask, cont_sorted, 0, 255, -1)
        angle_1 = self.get_angle_from_contour_mask(mask)
        print("angle1: {}".format(angle_1))

        mask = np.zeros_like(mask)
        cv2.drawContours(mask, cont_sorted, 1, 255, -1)
        angle_2 = self.get_angle_from_contour_mask(mask)
        print("angle2: {}".format(angle_2))

        angles = [angle_1, angle_2]
        # Get rid of the "Nones"
        angles = [ang for ang in angles if ang]

        avg_ang = 0.
        for ang in angles:
            avg_ang += ang / len(angles)

        # TODO what to do if both are bad
        img = self.draw_avg_line(img, avg_ang)

        # avg_ang = angle_1 + angle

        # minLineLength = 100
        # maxLineGap = 10
        # lines = cv2.HoughLinesP(mask,1,np.pi/180,100,minLineLength,maxLineGap)
        # print(len(lines))
        # for i in range(len(lines)):
            # for x1,y1,x2,y2 in lines[i]:
                # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


        return img


if __name__ == '__main__':
    seg = Segmenter()

    img_files = ["./example.jpg", "./example_hard.jpg"]
    for img_file in img_files:
        img = cv2.imread(img_file)
        seg_img = seg.segment_img(img)

        cv2.imshow("img", img)
        cv2.imshow("segmented", seg_img)
        cv2.waitKey(0)
