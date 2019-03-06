from IPython.core.debugger import set_trace
import os
import json
import pathlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

tf.enable_eager_execution()

## Data Import Functions ##
def preprocess_image(image, channels=3, size=[192,192]):
    image = tf.image.decode_image(image, channels=channels)
    image = tf.image.resize_images(image, size)
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path, channels):
    image = tf.read_file(path)
    return preprocess_image(image, channels)

def load_and_preprocess_command(path, outputs, omega_bins=15):
    command = json.load(open(path))
    result = []
    # Binning params
    for key in outputs:
        if key == 'omega':
            # Get one-hot encoding of omega bin
            omega = command[key]
            bin_size = 2.0/omega_bins
            bin_idx = int((omega+1)/bin_size)
            result.append([float(i == bin_idx) for i in range(omega_bins)])
        else:
            result.append(command[key])
    return result

def get_dataset(root, input_type='rgbd', outputs=['omega']):
    paths = [os.path.join(root, filename) for filename in os.listdir(root)
                if filename.endswith(('jpg', 'json'))]
    paths.sort()

    # Separate data into groups
    rgb_tensors = [load_and_preprocess_image(x, channels=3) for x in paths[2::3]]
    depth_tensors = [load_and_preprocess_image(x, channels=1) for x in paths[1::3]]
    if input_type == 'rgbd':
        input_tensors = tf.concat((rgb_tensors, depth_tensors), axis=3)
    elif input_type == 'rgb':
        input_tensors = rgb_tensors
    elif input_type == 'depth':
        input_tensors = depth_tensors
    else:
        raise ValueError('Invalid input type: \'', input_type, '\'')
    command_tensors = [load_and_preprocess_command(x, outputs) for x in paths[0::3]]
    return tf.data.Dataset.from_tensor_slices((input_tensors, command_tensors))

## Network Model ##
class DonkeyCarNet(tf.keras.Model):
    def __init__(self, output_size=1, omega_bins=15):
        super().__init__()
        self.output_size = output_size
        self.conv_layers = tf.keras.Sequential((
            Conv2D(24, (5,5), strides=2, activation='relu'),
            Conv2D(32, (5,5), strides=2, activation='relu'),
            Conv2D(64, (5,5), strides=2, activation='relu'),
            Conv2D(64, (3,3), strides=2, activation='relu'),
            Conv2D(64, (3,3), strides=1, activation='relu')
        ))
        self.linear_layers = tf.keras.Sequential((
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.1),
            Dense(50, activation='relu'),
            Dropout(0.1)
        ))
        self.omega_out_layer = Dense(omega_bins, activation='softmax', name='omega_out')
        if self.output_size == 2:
            self.vel_out_layer = Dense(1, activation='relu', name='vel_out')

    def call(self, inputs):
        x = self.conv_layers(inputs)
        x = self.linear_layers(x)
        omega_out = self.omega_out_layer(x)
        if self.output_size == 2:
            vel_out = self.vel_out_layer(x)
            return [omega_out, vel_out]
        else:
            return [omega_out]

if __name__=='__main__':
    # Model loading params
    load_model = False
    model_name = 'model'

    # Dataset
    test_name = 'blue_tape_lanes_vel_0_4'
    ws_root = os.getcwd().split('catkin_ws')[0]
    data_root = os.path.join(ws_root, 'data', test_name)
    outputs = ['omega']
    dataset = get_dataset(data_root, input_type='rgb', outputs=outputs)

    # Model
    model = DonkeyCarNet(len(outputs))
    model_file = os.path.join(data_root, model_name)
    if load_model and os.path.isfile(model_file + ".index"):
        model.load_weights(model_file)
        print("Loading model weights from ", model_file)
    optimizer = tf.train.AdamOptimizer(
        learning_rate = 0.001,
        beta1 = 0.9,
        beta2 = 0.99,
        epsilon = 1e-6
    )
    loss_function = 'categorical_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )

    # for data in dataset.take(1):
    #     inputs, outputs = data
    #     set_trace()
    #     y = model(inputs)

    # Training params
    batch_size = 10
    epochs = 50
    datapoints_per_epoch = 1
    # Fit the data
    history = model.fit(dataset.repeat(epochs).batch(batch_size),
              epochs=epochs*datapoints_per_epoch,
              steps_per_epoch=len(list(dataset))//(batch_size*datapoints_per_epoch),
              )


    model.save_weights(model_file)
    print("Saving model weights to ", model_file)

    losses = history.history['loss']
    plt.plot(losses)
    plt.show()
