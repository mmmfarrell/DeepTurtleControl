import json
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential, model_from_json

from data_generator import DataGenerator

"""
Note this only works for rgb inputs and continuous omega outputs right now.
"""


def get_model(img_size, outputs=1):
    input_shape = (img_size[1], img_size[0], 3)
    resnet_model = DenseNet201(weights=None, include_top=False,
                            input_shape=input_shape)

    x = layers.Flatten()(resnet_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    if outputs > 1:
        angle_out = layers.Dense(outputs, activation="softmax")(x)
    else:
        angle_out = layers.Dense(1, activation='linear')(x)

    transfer_model = Model(inputs=resnet_model.input, outputs=angle_out)

    return transfer_model

class CategoricalDistanceLoss:
    def __init__(self, outputs):
        self.outputs = outputs
        self.bin_width = 2.0/self.outputs
        self.sigma = self.bin_width*0.5

    def __call__(self, desired, predicted):
        ''' Compute a categorical loss, but include distance information

        Arguments:
            predicted: a list of probabilities for each category
            desired: a one-hot encoding of the correct category
        '''
        correct_idx = tf.cast(tf.reshape(tf.argmax(desired, axis=-1), (-1,1)), tf.float32)
        dlist = [(correct_idx-i)**2 * self.bin_width/self.sigma for i in range(self.outputs)]
        distances = tf.concat(dlist, axis=1)
        losses = -tf.exp(-distances)*tf.log(predicted)
        return tf.reduce_sum(losses)


def save_model_and_weights(model_file):
    print("Saving model weights to ", model_file)
    model.save_weights(model_file + ".h5")
    model_json = model.to_json()
    with open(model_file + '.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()


if __name__ == '__main__':
    # User Options
    crop_box = [0, 100, 480, 190]
    resize_scale = 3
    img_size = [int(x/resize_scale) for x in crop_box[2:]]
    gen_params = {'img_size': img_size,
                  'crop_box': crop_box,
                  'batch_size': 10,
                  'channels': 'rgb',
                  'shuffle': True,
                  'outputs': 5 } # outputs > 1 = omega_bins
    load_model = False
    train_epochs = 25
    # train_folders = ['tower_rope_circle_0_3_vel', 'tower_rope_circle_2_0_3_vel']
    train_folders = ['classical_bins_100']
    # train_folder = 'single_test'
    val_folder = 'tower_rope_circle_3_0_3_vel_val500'

    # Creat model
    model = get_model(gen_params['img_size'], outputs=gen_params['outputs'])

    # Get paths
    ws_root = os.getcwd().split('catkin_ws')[0]

    # Load model or make a directory for the model
    test_name = '&'.join(train_folders)
    model_dir_path = os.path.join(ws_root, 'saved_models', test_name)
    model_name = test_name + '_model'
    model_file = os.path.join(model_dir_path, model_name)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    if load_model:
        assert(os.path.isfile(model_file + ".h5")), \
            "Load model = True, but no model found at {}".format(model_file)
        # TODO currently this only loads the weights
        # The model itself can also be loaded with
        # json_file = open(model_file + ".json", "r")
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        print("Loading model weights from {}".format(model_file))
        model.load_weights(model_file + ".h5")

    # create train and validation data generators
    train_dir_paths = [os.path.join(ws_root, 'data', folder) for folder in train_folders]
    training_generator = DataGenerator(train_dir_paths, **gen_params)
    val_dir_path = os.path.join(ws_root, 'data', val_folder)
    val_generator = DataGenerator(val_dir_path, **gen_params)

    # Compile and train network
    if gen_params['outputs'] > 1:
        loss_function = CategoricalDistanceLoss(gen_params['outputs'])
    else:
        loss_function = 'mean_squared_error'
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=loss_function,
        metrics=['accuracy'])
    try:
        history = None
        history = model.fit_generator(generator=training_generator,
                                      validation_data=val_generator,
                                      epochs=train_epochs, use_multiprocessing=False,
                                      workers=1)
    except KeyboardInterrupt:
        '''If you exit with Ctrl + C the weights will be saved'''
        print("Got Keyboard Interrupt, saving model and closing")
    save_model_and_weights(model_file)

    if history:
        losses = history.history['loss']
        plt.plot(losses)
        plt.show()
