#!/usr/bin/env python

import numpy as np

import rospy
import cv2

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class TurtleControl():

    def __init__(self):
        # TODO
        # Cmd types: 0 - pure joy, 1 - constant vel, omega from joy, 2 - auto
        # Press
        # A - pure joy command
        # B - constant vel, omega joy command
        # X - autonomous command
        self.cmd_type = 0
        self.auto_cmd = None
        self.joy_vel = None
        self.joy_omega = None

        # TODO
        # Press C/D to start/stop recording
        self.record_data = False

        self.velocity_max = 1. 
        self.omega_max = 1.

        self._joy_sub = rospy.Subscriber("joy", Image,
                self.joy_callback)
        self._auto_cmd_sub = rospy.Subscriber("auto_cmd", TwistStamped,
                self.auto_cmd_callback)

        self._cmd_pub = rospy.Publisher('turtle_cmd', TwistStamped,
                queue_size=10)
        self._record_pub = rospy.Publisher('record_data', Bool,
                queue_size=10)

        rospy.spin()

    def publish_cmd(self):
        cmd_msg = TwistStamped
        cmd_msg.header.stamp = rospy.Time.now()
        self._cmd_pub.publish(cmd_msg)

    def publish_record(self):
        record_msg = Bool()
        record_msg.data = self.record_data

        self._record_pub.publish(record_msg)

    def joy_callback(self, msg):
        self.joy_vel = (msg.blah - 1500.) / 500. # gives -1. to 1.
        self.joy_omega = (msg.blah - 1500.) / 500. # gives -1. to 1.

        self.joy_vel = self.velocity_max * self.joy_vel
        self.joy_omega = self.velocity_max * self.joy_omega

        if pressA:
            self.cmd_type = 0
        elif pressB:
            self.cmd_type = 1
        elif pressC:
            self.cmd_type = 2
        elif pressD:
            self.record_data = True
            self.publish_record()
        elif pressE:
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

