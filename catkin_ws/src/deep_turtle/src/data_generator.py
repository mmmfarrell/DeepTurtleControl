import numpy as np
import tensorflow.keras as keras
import json
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

"""
Note this only works for rgb inputs and continuous omega outputs right now.
"""

def get_and_preprocess_img(img_path, img_size):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

def get_and_preprocess_command(cmd_path):
    command = json.load(open(cmd_path))

    return command['omega']

def get_sorted_files_in_dir_with(directory, string_in_filename):
    paths = [os.path.join(directory, filename) for filename in
        os.listdir(directory) if (filename.endswith(('jpg', 'json')) and
                        string_in_filename in filename)]
    paths.sort()

    return paths

class DataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras. This is taken from the example.
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, directory, channels='rgb', batch_size=32, img_size=(32,32),
                 shuffle=True):
        '''Initialization'''
        self.directory = directory
        self.channels = channels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

        self.rgb_img_paths = get_sorted_files_in_dir_with(directory, 'rgb')
        self.command_json_paths = get_sorted_files_in_dir_with(directory,
                'commands')

        # Make sure channels is set correctly
        assert(channels=='rgb' or channels=='rgbd')
        self.use_depth = channels == 'rgbd'

        # Check that we have same number of data points for each type we need.
        assert(len(self.rgb_img_paths) == len(self.command_json_paths))

        if self.use_depth:
            self.depth_img_paths = get_sorted_files_in_dir_with(directory,
                'depth')
            assert(len(self.rgb_img_paths) == len(self.depth_img_paths))

        self.total_num_samples = len(self.rgb_img_paths)
        self.sample_indeces = np.arange(self.total_num_samples)

        # Call to reset/setup current data
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(self.total_num_samples / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        batch_indexes = self.sample_indeces[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(batch_indexes)

        return X, Y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.sample_indeces)

    def __data_generation(self, batch_indexes):
        '''Generates data containing batch_size samples''' 
        # Initialization
        num_channels = len(self.channels)

        X = np.empty((self.batch_size, *self.img_size, num_channels))
        Y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, img_idx in enumerate(batch_indexes):

            # Get preprocessed input
            assert(not self.use_depth), "Don't know how to preprocess depth yet"
            img_path = self.rgb_img_paths[img_idx]
            img_input = get_and_preprocess_img(img_path, self.img_size)
            X[i,] = img_input

            # Get output for the img
            cmd_path = self.command_json_paths[img_idx]
            command_output = get_and_preprocess_command(cmd_path)
            Y[i] = command_output

        return X, Y
