#!/usr/bin/env python
from __future__ import print_function

import numpy as np

import os
import cv2
import json

class Segmenter():

    def __init__(self):
        pass

    def get_angle_from_contour_mask(self, img, mask):
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
        if avg_angle:
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
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        return img

    def filter_angles(self, angles):
        # Filter out the "Nones"
        angles = [ang for ang in angles if ang]

        # Wrap between -pi/2, pi/2
        for idx, ang in enumerate(angles):
            # if ang < -np.pi / 2.:
                # ang += 2. * np.pi
            if ang > np. pi /2.:
                ang -=  np.pi
            angles[idx] = ang

        return angles

    # def perspective_warp(self, img):
        # dst_size=(640, 480)
        # src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
        # dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
        # img_size = np.float32([(img.shape[1],img.shape[0])])

        # src = src* img_size
        # # For destination points, I'm arbitrarily choosing some points to be
        # # a nice fit for displaying our warped result
        # # again, not exact, but close enough for our purposes
        # dst = dst * np.float32(dst_size)
        # # Given src and dst points, calculate the perspective transform matrix
        # M = cv2.getPerspectiveTransform(src, dst)
        # # Warp the image using OpenCV warpPerspective()
        # warped = cv2.warpPerspective(img, M, dst_size)
        # cv2.imshow("warped", warped)

        # return warped


    def get_poly_from_mask(self, mask):
        ydata, xdata = np.where(mask == 255)

        z = np.polyfit(ydata, xdata, 3)
        f = np.poly1d(z)

        return f

    def draw_poly(self, f, img):
        y_fit = np.arange(0, img.shape[0], 1)
        x_fit = f(y_fit)

        x_fit = map(int, x_fit)
        # print("xfit: ", map(int, x_fit))
        # print("yfit: ", y_fit)

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

        y_middle = img.shape[0] / 3
        x_middle = f(y_middle)
        x_middle = int(x_middle)
        cv2.circle(img, (x_middle, y_middle), 10, (0, 255, 0), -1)
        # cv2.imshow("fit", fit)
        # print(xdata)
        # print(ydata)

    def segment_img(self, img):

        # self.perspective_warp(img)

        # Hue Lightness Saturation
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        margin = 25
        lower_white = np.array([0, 255 - margin, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(img_hls, lower_white, upper_white)
        # cv2.imshow("mask", mask)

        i, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        print(len(cont_sorted))
        angles = []
        for idx, cont in enumerate(cont_sorted):
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, cont_sorted, idx, 255, -1)
            ang = self.get_angle_from_contour_mask(img, mask)
            # print("angle1: {}".format(angle_1))
            angles.append(ang)
            cv2.imshow("cont " + str(idx), mask)

            print("Contour area: ", cv2.contourArea(cont_sorted[idx]))
            cont_area = cv2.contourArea(cont_sorted[idx])
            cont_thresh = 4000.
            if cont_area > cont_thresh:
                poly = self.get_poly_from_mask(mask)
                self.draw_poly(poly, img)

            # mask = np.zeros_like(mask)
            # cv2.drawContours(mask, cont_sorted, 1, 255, -1)
            # angle_2 = self.get_angle_from_contour_mask(img, mask)
            # print("angle2: {}".format(angle_2))

        # angles = [angle_1, angle_2]
        angles = self.filter_angles(angles)
        print(angles)

        avg_ang = 0.
        for ang in angles:
            avg_ang += ang / len(angles)

        # TODO what to do if both are bad
        img = self.draw_avg_line(img, avg_ang)

        return img


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
