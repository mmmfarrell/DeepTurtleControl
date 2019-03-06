import os
import json
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

tf.enable_eager_execution()

## Data Import Functions ##
def preprocess_image(image, channels=3, size=[120,120]):
    image = tf.image.decode_image(image, channels=channels)
    print(image.shape)
    print(image.dtype)
    image = tf.image.resize_images(image, size)
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path, channels):
    image = tf.read_file(path)
    return preprocess_image(image, channels)

def load_and_preprocess_command(path, outputs):
    command = json.load(open(path))
    return [command[key] for key in outputs]

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
def get_model(output_size=1):
    model = tf.keras.Sequential()

    # Strided 5x5 Convolutions
    for output_channels in [24,36,48]:
        model.add(layers.Conv2D(filters=output_channels,
            kernel_size=5, strides=2, activation='relu'))

    # Non-strided 3x3 Convolutions
    for output_channels in [64,64]:
        model.add(layers.Conv2D(filters=output_channels, kernel_size=3,
            activation='relu'))

    # Fully-connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(output_size))

    return model

if __name__=='__main__':
    # Model loading params
    load_model = False
    model_name = 'model'

    # Dataset
    test_name = 'blue_tape_vel_0_4'
    ws_root = os.getcwd().split('catkin_ws')[0]
    data_root = os.path.join(ws_root, 'data', test_name)
    outputs = ['omega']
    # dataset = get_dataset(data_root, input_type='rgbd', outputs=outputs)
    dataset = get_dataset(data_root, input_type='rgb', outputs=outputs)

    # Model
    model = get_model(output_size=len(outputs))

    model_file = os.path.join(data_root, model_name)
    if load_model and os.path.isfile(model_file + ".index"):
        model.load_weights(model_file)
        print("Loading model weights from ", model_file)
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='mean_squared_error'
    )

    # Training params
    batch_size = 10
    epochs = 30
    # Fit the data
    history = model.fit(dataset.batch(batch_size).repeat(epochs),
              epochs=epochs, steps_per_epoch=len(list(dataset))//batch_size)

    losses = history.history['loss']
    plt.plot(losses)
    plt.show()

    model.save_weights(model_file)
    print("Saving model weights to ", model_file)
