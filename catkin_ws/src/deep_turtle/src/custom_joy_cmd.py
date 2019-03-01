#!/usr/bin/env python

import numpy as np

import rospy
import cv2

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

class TurtleControl():

    def __init__(self):
        # Cmd types: 0 - pure joy, 1 - constant vel, omega from joy, 2 - auto
        # Press
        # A - pure joy command
        # B - constant vel, omega joy command
        # Y - autonomous command
        self.cmd_type = 0
        self.cmd_msg = Twist()
        self.auto_cmd = Twist()
        self.joy_vel = 0.
        self.joy_omega = 0.

        # Press C/D to start/stop recording
        self.record_data = False

        self.velocity_max = rospy.get_param("~joy_velocity_max")
        self.omega_max = rospy.get_param("~joy_omega_max")
        self.constant_vel = rospy.get_param("~constant_vel")

        self._joy_sub = rospy.Subscriber("joy", Joy,
                self.joy_callback)
        self._auto_cmd_sub = rospy.Subscriber("auto_cmd", Twist,
                self.auto_cmd_callback)

        self._cmd_pub = rospy.Publisher('turtle_cmd', Twist,
                queue_size=10)
        self._record_pub = rospy.Publisher('record_data', Bool,
                queue_size=10)

        self._timer = rospy.Timer(rospy.Duration(0.05), self.timer_callback)

        rospy.spin()

    def timer_callback(self, event):
        # self.cmd_msg.header.stamp = rospy.Time.now()
        if self.cmd_type == 0: # pure joy command
            self.cmd_msg.linear.x = self.joy_vel
            self.cmd_msg.angular.z = self.joy_omega
        elif self.cmd_type == 1: # omega only command
            self.cmd_msg.linear.x = self.constant_vel
            self.cmd_msg.angular.z = self.joy_omega
        elif self.cmd_type == 2: # auto command
            self.cmd_msg.linear.x = self.auto_cmd.linear.x
            self.cmd_msg.angular.z = self.auto_cmd.angular.z
        self._cmd_pub.publish(self.cmd_msg)

    def publish_record(self):
        record_msg = Bool()
        record_msg.data = self.record_data

        self._record_pub.publish(record_msg)

    def joy_callback(self, msg):
        self.joy_vel = msg.axes[1] # from -1 to 1
        self.joy_omega = msg.axes[0] # from -1 to 1

        self.joy_vel = self.velocity_max * self.joy_vel
        self.joy_omega = self.omega_max * self.joy_omega

        if msg.buttons[0]: # A Button
            self.cmd_type = 0 # pure joy command
            rospy.loginfo_throttle(0.5, "Pure Joy Command")
        elif msg.buttons[1]: # B Button
            self.cmd_type = 1 # omega only command
            rospy.loginfo_throttle(0.5, "Constant Vel Command")
        elif msg.buttons[3]: # Y Button
            self.cmd_type = 2 # autonomous command
            rospy.loginfo_throttle(0.5, "Autonomous Command")
        elif msg.buttons[7]: # Start Button
            self.record_data = True
            rospy.loginfo_throttle(0.5, "Start Recording")
            self.publish_record()
        elif msg.buttons[6]: # Back button
            rospy.loginfo_throttle(0.5, "Stop Recording")
            self.record_data = False
            self.publish_record()

    def auto_cmd_callback(self, msg):
        self.auto_cmd = msg

if __name__ == '__main__':
    rospy.init_node('turtle_control', anonymous=True)
    try:
        turtle_control = TurtleControl()
    except:
        rospy.ROSInterruptException
    pass

