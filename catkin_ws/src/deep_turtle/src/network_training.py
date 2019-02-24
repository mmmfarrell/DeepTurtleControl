import os
import json
import pathlib
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

## Data Import Functions ##
def preprocess_image(image, size=[192,192]):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize_images(image, size)
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    print(path)
    image = tf.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_command(path):
    command = json.load(open(path))
    return [command['v'], command['omega']]

def get_dataset(root):
    paths = [os.path.join(root, filename) for filename in os.listdir(root)]
    paths.sort()

    # Separate data into groups
    rgb_tensors = [load_and_preprocess_image(x) for x in paths[2::3]]
    depth_tensors = [load_and_preprocess_image(x) for x in paths[1::3]]
    command_tensors = [load_and_preprocess_command(x) for x in paths[0::3]]
    return tf.data.Dataset.from_tensor_slices((rgb_tensors, depth_tensors, command_tensors))

## Network Model ##

if __name__=='__main__':
    test_name = 'rgbd'
    ws_root = os.getcwd().split('catkin_ws')[0]
    data_root = os.path.join(ws_root, 'data', test_name)

    dataset = get_dataset(data_root)
    for i,x in enumerate(dataset.take(1)):
        rgb, depth, (v, omega) = x
