#!/usr/bin/env python

import numpy as np

import rospy
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped

from cv_bridge import CvBridge, CvBridgeError

class DataRecorder():

    def __init__(self):
        self.bridge = CvBridge()

        self._image_sub = rospy.Subscriber("/camera/color/image_raw", Image,
                self.image_callback)

        rospy.spin()

    def image_callback(self, msg):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
          print(e)
          return

        cv2.imwrite("/home/mmmfarrell/test.jpg", cv_image)


        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

if __name__ == '__main__':
    rospy.init_node('data_recorder', anonymous=True)
    try:
        data_record = DataRecorder()
    except:
        rospy.ROSInterruptException
    pass

