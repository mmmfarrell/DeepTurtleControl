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

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class ContinuousNeuralController():

    def __init__(self):
        self.record = False
        self.current_record_idx = 0

        self.bridge = CvBridge()

        print("b4 param")
        # self.test_name = rospy.get_param("~model_name")
        self.test_name = "rope1/rope1"

        rospack = rospkg.RosPack()
        ws_root = rospack.get_path("deep_turtle").split('catkin_ws')[0]
        model_dir = os.path.join(ws_root , 'saved_models', self.test_name)

        self.model = self.load_model(model_dir)
        self.graph = tf.get_default_graph()

        self.last_command = 0.

        self._rgb_sub = rospy.Subscriber("rgb_image", Image,
                self.rgb_image_callback, queue_size=1, buff_size=2**24)
        self._cmd_pub = rospy.Publisher('auto_cmd_raw', Twist,
                queue_size=10)
        self._cmd_smooth_pub = rospy.Publisher('auto_cmd_smooth', Twist,
                queue_size=10)

        rospy.spin()

    def load_model(self, model_file):
        # assert(os.path.isfile(model_file + ".h5")), \
            # "No model found at {}".format(model_file + ".h5")
        # assert(os.path.isfile(model_file + ".json")), \
            # "No model found at {}".format(model_file + ".json")
        if (not os.path.isfile(model_file + ".json")):
            rospy.logwarn("ERROR")
        else:
            rospy.logwarn("GOOD")
        # load the model itself
        json_file = open(model_file + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load the weights
        print("Loading model weights from {}".format(model_file))
        model.load_weights(model_file + ".h5")

        return model


    def rgb_image_callback(self, rgb_msg):
        try:
          rgb_cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
          output = self.predict_from_cv_img(rgb_cv_image)
          self.publish_cmd(output)
          self.publish_smooth_cmd(output)
        except CvBridgeError as e:
          print(e)
          return

        # self.current_record_idx += 1

    def predict_from_cv_img(self, img):
        x = cv2.resize(img, (120, 120))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        with self.graph.as_default():
            out = self.model.predict(x)
        return out[0][0]

    def publish_cmd(self, predict_out):
        cmd_msg = Twist()
        cmd_msg.angular.z = predict_out
        self._cmd_pub.publish(cmd_msg)

    def publish_smooth_cmd(self, predict_out):
        cmd_msg = Twist()
        alpha = 0.5
        cmd_msg.angular.z = alpha * self.last_command + (1. - alpha) * predict_out
        self._cmd_smooth_pub.publish(cmd_msg)


if __name__ == '__main__':
    rospy.init_node('neural_controller', anonymous=True)
    try:
        controller = ContinuousNeuralController()
    except:
        rospy.ROSInterruptException
    pass

