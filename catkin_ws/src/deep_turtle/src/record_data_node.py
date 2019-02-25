#!/usr/bin/env python

import numpy as np

import rospy
import cv2
import json

import message_filters

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped

from cv_bridge import CvBridge, CvBridgeError

class DataRecorder():

    def __init__(self):
        self.record = False
        self.current_record_idx = 0

        self.bridge = CvBridge()

        # self._image_sub = rospy.Subscriber("/camera/color/image_raw", Image,
                # self.image_callback)
        # self._image_sub = rospy.Subscriber("/camera/color/image_raw", Image,
                # self.image_callback)
        # self._image_sub = rospy.Subscriber("/camera/color/image_raw", Image,
                # self.image_callback)

        # TODO if time stamps are all good, this is the right way to do this
        rgb_sub = message_filters.Subscriber('rgb_image', Image)
        depth_sub = message_filters.Subscriber('depth_image', Image)
        cmd_sub = message_filters.Subscriber('turtlebot_command', TwistStamped)

        ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, cmd_sub], 10)
        ts.registerCallback(callback)

        rospy.spin()

    def combined_callback(self, rgb_msg, depth_msg, cmd_msg):
        index_string = str(self.current_record_idx).zfill(5)

        data = {}
        data['v'] = vel
        data['omega'] = vel
        with open('data.txt', 'w') as outfile:  
            json.dump(data, outfile)

        index_string += 1



    # def image_callback(self, msg):
        # try:
          # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # except CvBridgeError as e:
          # print(e)
          # return

        # if self.record:
            # cv2.imwrite("/home/mmmfarrell/test.jpg", cv_image)


        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)

if __name__ == '__main__':
    rospy.init_node('data_recorder', anonymous=True)
    try:
        data_record = DataRecorder()
    except:
        rospy.ROSInterruptException
    pass

