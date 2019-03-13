from IPython.core.debugger import set_trace
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

tf.enable_eager_execution()

## Data Import Functions ##
def load_and_preprocess_image(path, channels, size=[192,192], crop_box=[280, 0, 192, 640],
                    augment_N=10, random_crop_margin=[30,30], flip=None):
    image = tf.read_file(path)
    image = tf.image.decode_image(image, channels=channels)
    image = tf.image.crop_to_bounding_box(image, *crop_box)
    image = tf.image.resize_images(image, size)
    image /= 255.0  # normalize to [0,1] range

    # Create augmentations
    images = []
    random_crop_box = [size[i] - random_crop_margin[i] for i in range(2)] + [channels]
    for i in range(augment_N):
        augment_img = image
        augment_img = tf.image.random_crop(augment_img, random_crop_box)
        if flip is not None and flip[i] == 1:
            augment_img = tf.image.flip_left_right(augment_img)
        images.append(augment_img)
    return images

def load_and_preprocess_command(path, outputs, omega_bins=15, augment_N=10, flip=None):
    bin_size = 2.0/omega_bins
    command = json.load(open(path))
    results = []
    for i in range(augment_N):
        result = []
        # Binning params
        for key in outputs:
            if key == 'omega':
                # Get one-hot encoding of omega bin
                omega = command[key] * flip[i]*-1 # Negate omega for flipped images
                bin_idx = int((omega+1)/bin_size)
                result.append([float(i == bin_idx) for i in range(omega_bins)])
            else:
                result.append(command[key])
        results.append(result)
    return results

def get_dataset(roots, input_type='rgbd', outputs=['omega'], augment_N=10):
    paths = []
    for root in roots:
        paths.extend([os.path.join(root, filename) for filename in os.listdir(root)
                    if filename.endswith(('jpg', 'json'))])
    paths.sort()

    # Separate data into groups
    rgb_tensors = []
    depth_tensors = []
    command_tensors = []
    for i in range(0, len(paths), 3):
        print('Creating dataset: {:.1f}%'.format(i*100/len(paths)), end='\r')
        rgb_path = paths[i+2]
        depth_path = paths[i+1]
        command_path = paths[i]
        random_flip = np.random.randint(0, 2, augment_N)
        rgb_tensors.extend(load_and_preprocess_image(rgb_path, channels=3, augment_N=augment_N, flip=random_flip))
        depth_tensors.extend(load_and_preprocess_image(depth_path, channels=1, augment_N=augment_N, flip=random_flip))
        command_tensors.extend(load_and_preprocess_command(command_path, outputs=outputs, augment_N=augment_N, flip=random_flip))
    if input_type == 'rgbd':
        input_tensors = tf.concat((rgb_tensors, depth_tensors), axis=3)
    elif input_type == 'rgb':
        input_tensors = rgb_tensors
    elif input_type == 'depth':
        input_tensors = depth_tensors
    else:
        raise ValueError('Invalid input type: \'', input_type, '\'')
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
    # Dataset
    test_folders = ['blue_tape_lanes_vel_0_4']
    ws_root = os.getcwd().split('catkin_ws')[0]
    data_roots = [os.path.join(ws_root, 'data', test_name) for test_name in test_folders]
    outputs = ['omega']
    dataset = get_dataset(data_roots, input_type='rgb', outputs=outputs)

    # Model
    load_model = False
    model_name = '&'.join(test_folders) + '_model'
    model = DonkeyCarNet(len(outputs))
    model_file = os.path.join(data_roots[0], model_name)
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
