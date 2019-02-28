from IPython.core.debugger import set_trace
import os
import json
import pathlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

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

def get_dataset(root, batch_size=5, shuffle=True):
    paths = [os.path.join(root, filename) for filename in os.listdir(root)]
    paths.sort()

    # Separate data into groups
    rgb_tensors = [load_and_preprocess_image(x) for x in paths[2::3]]
    depth_tensors = [load_and_preprocess_image(x) for x in paths[1::3]]
    command_tensors = [load_and_preprocess_command(x) for x in paths[0::3]]
    ds = tf.data.Dataset.from_tensor_slices((rgb_tensors, depth_tensors, command_tensors))
    set_trace()
    # TODO: Add shuffle
    return ds.batch(batch_size)

## Network Model ##
def get_model(outputs=1):
    model = tf.keras.Sequential()

    # Strided 5x5 Convolutions
    for output_channels in [24,36,48]:
        model.add(layers.Conv2D(filters=output_channels, kernel_size=5, strides=2, activation='relu'))

    # Non-strided 3x3 Convolutions
    for output_channels in [64,64]:
        model.add(layers.Conv2D(filters=output_channels, kernel_size=3, activation='relu'))

    # Fully-connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(outputs))

    return model

if __name__=='__main__':
    # Dataset
    test_name = 'rgbd'
    ws_root = os.getcwd().split('catkin_ws')[0]
    data_root = os.path.join(ws_root, 'data', test_name)
    dataset = get_dataset(data_root)

    # Model
    num_outputs = 1
    model = get_model(num_outputs)

    # Training params
    loss_function = tf.keras.losses.MSE
    num_epochs = 1

    for epoch in range(num_epochs):
        for i,x in enumerate(dataset):
            rgb, depth, command = x

            # Concatenate rgb and depth
            rgbd = tf.concat([rgb, depth], 2)
            print(rgbd.shape)

            # Estimate command
            command_pred = model(rgbd)

            # Compute Loss
            loss = loss_function(command, command_pred)

            # Update model
