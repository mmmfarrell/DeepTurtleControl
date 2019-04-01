import json
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.callbacks import EarlyStopping

from data_generator import DataGenerator

"""
Note this only works for rgb inputs and continuous omega outputs right now.
"""


def get_model(img_size):
    input_shape = (img_size[0], img_size[1], 3)
    resnet_model = ResNet50(weights='imagenet', include_top=False,
                            input_shape=input_shape)

    x = layers.Flatten()(resnet_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    angle_out = layers.Dense(1, activation='linear')(x)

    transfer_model = Model(inputs=resnet_model.input, outputs=angle_out)

    return transfer_model


def save_model_and_weights(model_file):
    print("Saving model weights to ", model_file)
    model.save_weights(model_file + ".h5")
    model_json = model.to_json()
    with open(model_file + '.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()


if __name__ == '__main__':
    # User Options
    gen_params = {'img_size': (120, 120),
                  'batch_size': 16,
                  'channels': 'rgb',
                  'shuffle': True}
    load_model = True
    train_epochs = 100
    train_folders = ['tower_rope_circle_0_3_vel', 'tower_rope_circle_2_0_3_vel']
    val_folders = ['tower_rope_circle_3_0_3_vel']

    # Creat model
    model = get_model(gen_params['img_size'])

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
    val_dir_paths = [os.path.join(ws_root, 'data', folder) for folder in val_folders]
    val_generator = DataGenerator(val_dir_paths, **gen_params)

    # Compile and train network
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='mean_squared_error')

    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    try:
        history = None
        history = model.fit_generator(generator=training_generator,
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=25,
                                      epochs=train_epochs, use_multiprocessing=True,
                                      workers=4)
    except KeyboardInterrupt:
        '''If you exit with Ctrl + C the weights will be saved'''
        print("Got Keyboard Interrupt, saving model and closing")
    save_model_and_weights(model_file)

    if history:
        losses = history.history['loss']
        plt.plot(losses)
        plt.show()

