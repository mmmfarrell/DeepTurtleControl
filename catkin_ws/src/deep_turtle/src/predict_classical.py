#!/usr/bin/env python

import numpy as np

import os
import rospy
import rospkg
import cv2
import json

import message_filters

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from cv_bridge import CvBridge, CvBridgeError

from classical.segment_ropes import Segmenter

class ContinuousNeuralController():

    def __init__(self):
        self.record = False
        self.current_record_idx = 0

        self.bridge = CvBridge()

        print("Classical Control")

        rospack = rospkg.RosPack()

        self.seg = Segmenter()

        self.last_command = 0.

        self._cmd_pub = rospy.Publisher('auto_cmd_raw', Twist,
                queue_size=10)
        self._cmd_smooth_pub = rospy.Publisher('auto_cmd_smooth', Twist,
                queue_size=10)
        self._mask_pub = rospy.Publisher('seg_mask', Image)

        self._rgb_sub = rospy.Subscriber("rgb_image", Image,
                self.rgb_image_callback, queue_size=1, buff_size=2**24)

        rospy.spin()

    def rgb_image_callback(self, rgb_msg):
        try:
          rgb_cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
          seg_img = rgb_cv_image.copy()
          seg_img, mid_x = self.seg.segment_img(seg_img)
          omega = self.compute_control(mid_x)
          # print(omega)

          # cv2.imshow("raw img", rgb_cv_image)
          # cv2.imshow("output", seg_img)
          # cv2.waitKey(1)
          # output = self.predict_from_cv_img(rgb_cv_image)
          self.publish_cmd(omega)
          self.publish_smooth_cmd(omega)

          image_msg = self.bridge.cv2_to_imgmsg(seg_img, encoding="bgr8")
          self._mask_pub.publish(image_msg)
        except CvBridgeError as e:
          print(e)
          return

        # self.current_record_idx += 1

    def compute_control(self, mid_x):
        scale = 2.0
        #true_middle = 320
        # This is only because the camera isnt aligned with the robot, its tilted
        true_middle = 240
        max_err = float(true_middle)

        # If err is positive, turn left, which is positive omega
        err = true_middle - mid_x

        omega = scale * err / max_err

        return omega


    def publish_cmd(self, predict_out):
        cmd_msg = Twist()
        cmd_msg.angular.z = predict_out
        self._cmd_pub.publish(cmd_msg)

    def publish_smooth_cmd(self, predict_out):
        cmd_msg = Twist()
        alpha = 0.5
        cmd_msg.linear.x = 0.3
        cmd_msg.angular.z = alpha * self.last_command + (1. - alpha) * predict_out
        self._cmd_smooth_pub.publish(cmd_msg)


if __name__ == '__main__':
    rospy.init_node('neural_controller', anonymous=True)
    try:
        controller = ContinuousNeuralController()
    except:
        rospy.ROSInterruptException
    pass

