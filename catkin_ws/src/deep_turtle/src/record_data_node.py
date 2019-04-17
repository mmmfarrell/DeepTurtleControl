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

class DataRecorder():

    def __init__(self):
        self.record = False
        self.current_record_idx = 0

        self.bridge = CvBridge()

        self.test_name = rospy.get_param("~test_name")

        self.frame_counter = 0
        self.record_n_frames = rospy.get_param("~record_n_frames")

        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path("deep_turtle")
        self.record_dir = self.package_path + "/data/" + self.test_name
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        else:
            print("Found directory: {}".format(self.record_dir))
            files = sorted(os.listdir(self.record_dir))
            if (len(files)):
              last_file = files[-1]
              self.current_record_idx = int(last_file[0:6]) + 1
              print("Starting recording at # {}".format(self.current_record_idx))

        self._record_sub = rospy.Subscriber("record_data", Bool,
                self.record_callback)

        rgb_sub = message_filters.Subscriber('rgb_image', Image)
        # depth_sub = message_filters.Subscriber('depth_image', Image)
        cmd_sub = message_filters.Subscriber('turtlebot_command', Twist)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub,
            cmd_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.combined_callback)

        rospy.spin()

    def combined_callback(self, rgb_msg, cmd_msg):
        if not self.record:
            return

        if (self.frame_counter % self.record_n_frames) == 0:
            self.frame_counter += 1
        else:
            self.frame_counter += 1
            return

        index_string = str(self.current_record_idx).zfill(5)

        # Write Commands as Json
        json_file_name = self.record_dir + "/" + index_string + "_commands.json"
        json_data = {}
        json_data['v'] = cmd_msg.linear.x
        json_data['omega'] = cmd_msg.angular.z
        with open(json_file_name, 'w') as json_file:  
            json.dump(json_data, json_file)

        # Write rgb image as jpg
        rgb_file_name = self.record_dir + "/" + index_string + "_rgb.jpg"
        try:
          rgb_cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
          print(e)
          return
        cv2.imwrite(rgb_file_name, rgb_cv_image)

        self.current_record_idx += 1

    def record_callback(self, msg):
        self.record = msg.data


if __name__ == '__main__':
    rospy.init_node('data_recorder', anonymous=True)
    try:
        data_record = DataRecorder()
    except:
        rospy.ROSInterruptException
    pass

