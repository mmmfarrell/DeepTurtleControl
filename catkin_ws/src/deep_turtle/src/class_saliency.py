#!/usr/bin/env python

import numpy as np

import os
import cv2
import json

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras import backend


def load_model(model_file):
    # assert(os.path.isfile(model_file + ".h5")), \
        # "No model found at {}".format(model_file + ".h5")
    # assert(os.path.isfile(model_file + ".json")), \
        # "No model found at {}".format(model_file + ".json")
    # load the model itself
    json_file = open(model_file + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load the weights
    print("Loading model weights from {}".format(model_file))
    model.load_weights(model_file + ".h5")

    return model

def get_img_crop(img):
    crop_box = [0, 100, 640, 190]
    resize_scale = 3
    img_size = [int(x/resize_scale) for x in crop_box[2:]]

    x, y, w, h = crop_box
    img = img[y:y+h, x:x+w, :]
    img = cv2.resize(img, tuple(img_size))
    # cv2.imshow("cropped", img)
    return img

def get_preprocessed_input(img):
    img = get_img_crop(img)
    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)
    # cv2.imshow("preprocessed:", img[0, :, :, :])

    return img

def predict_from_cv_img(model, img):
    img = get_preprocessed_input(img)

    # with self.graph.as_default():
    out = model.predict(img)

    # print("network out: ", out)
    bin_num = np.argmax(out)

    num_bins = 5
    bin_size = 2./num_bins
    cmd_out = bin_num * bin_size + bin_size/2. - 1.
    # print("cmd out: ", cmd_out)
    return cmd_out

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    backend.set_session(sess)

    # Load the model
    model_name = 'class_bins_w_human/classical_bins_model'
    ws_root = '/home/mmmfarrell/DeepTurtleControl/'
    model_dir = os.path.join(ws_root , 'saved_models', model_name)
    model = load_model(model_dir)

    # Load example img
    # img_path = '/home/mmmfarrell/DeepTurtleControl/data/classical_bins/00000_rgb.jpg'
    # true_bin = 2
    # true_bin = 4
    # img_path = '/home/mmmfarrell/DeepTurtleControl/data/classical_bins/00105_rgb.jpg'
    # true_bin = 3
    # img_path = '/home/mmmfarrell/DeepTurtleControl/data/michael_bins_outside/00105_rgb.jpg'
    # img_path = '/home/mmmfarrell/DeepTurtleControl/data/michael_bins_outside/00000_rgb.jpg'
    img_path = '/home/mmmfarrell/DeepTurtleControl/data/michael_bins/00100_rgb.jpg'
    img = cv2.imread(img_path)
    raw_crop_img = get_img_crop(img)

    input_img = get_preprocessed_input(img)

    out = model.predict(input_img)
    print(out)
    print(np.argmax(out))
    true_bin = np.argmax(out)

    # Evaluate the gradient of the classification w.r.t. input. Evaluate at
    # current input image
    input_tensor = model.input
    # output_tensor = model.output
    output_tensor = model.output[:, true_bin]
    gradients = backend.gradients(output_tensor, input_tensor)

    evaluated_gradients = sess.run(gradients, feed_dict={model.input:input_img})

    # Normalize the gradients so we can see them
    abs_grad = np.abs(evaluated_gradients[0])
    sum_grad = np.sum(abs_grad)
    normal_grad = abs_grad / sum_grad

    gray_grad = np.sum(normal_grad[0, :], axis=2)
    norm_gray = gray_grad / gray_grad.max()

    # Draw the gradients and the raw img
    norm_color = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)
    raw_crop_img = raw_crop_img.astype(float)/ 255.
    stacked = np.vstack((raw_crop_img, norm_color))
    cv2.imshow("Class Saliency", stacked)
    cv2.waitKey(0)
    cv2.imwrite("./class_saliency1.jpg", stacked)
