import os
import json
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential

from data_generator import DataGenerator

## Data Import Functions ##
# def preprocess_image(image, channels=3, size=[120,120]):
    # image = tf.image.decode_image(image, channels=channels)
    # print(image.shape)
    # print(image.dtype)
    # image = tf.image.resize_images(image, size)
    # image /= 255.0  # normalize to [0,1] range
    # return image

# def load_and_preprocess_image(path, channels):
    # image = tf.read_file(path)
    # return preprocess_image(image, channels)

# def load_and_preprocess_command(path, outputs):
    # command = json.load(open(path))
    # return [command[key] for key in outputs]

# def get_dataset(root, input_type='rgbd', outputs=['omega']):
    # paths = [os.path.join(root, filename) for filename in os.listdir(root)
                # if filename.endswith(('jpg', 'json'))]
    # paths.sort()

    # # Separate data into groups
    # rgb_tensors = [load_and_preprocess_image(x, channels=3) for x in paths[2::3]]
    # depth_tensors = [load_and_preprocess_image(x, channels=1) for x in paths[1::3]]
    # if input_type == 'rgbd':
        # input_tensors = tf.concat((rgb_tensors, depth_tensors), axis=3)
    # elif input_type == 'rgb':
        # input_tensors = rgb_tensors
    # elif input_type == 'depth':
        # input_tensors = depth_tensors
    # else:
        # raise ValueError('Invalid input type: \'', input_type, '\'')
    # command_tensors = [load_and_preprocess_command(x, outputs) for x in paths[0::3]]
    # return tf.data.Dataset.from_tensor_slices((input_tensors, command_tensors))

# ## Network Model ##
# def get_model(output_size=1):
    # model = tf.keras.Sequential()

    # # Strided 5x5 Convolutions
    # for output_channels in [24,36,48]:
        # model.add(layers.Conv2D(filters=output_channels,
            # kernel_size=5, strides=2, activation='relu'))

    # # Non-strided 3x3 Convolutions
    # for output_channels in [64,64]:
        # model.add(layers.Conv2D(filters=output_channels, kernel_size=3,
            # activation='relu'))

    # # Fully-connected layers
    # model.add(layers.Flatten())
    # model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dense(50, activation='relu'))
    # model.add(layers.Dense(output_size))

    # return model

if __name__=='__main__':
    # # Model loading params
    # load_model = False
    # model_name = 'model'

    # # Dataset
    # test_name = 'blue_tape_vel_0_4'
    # ws_root = os.getcwd().split('catkin_ws')[0]
    # data_root = os.path.join(ws_root, 'data', test_name)
    # outputs = ['omega']
    # # dataset = get_dataset(data_root, input_type='rgbd', outputs=outputs)
    # dataset = get_dataset(data_root, input_type='rgb', outputs=outputs)

    # # Model
    # model = get_model(output_size=len(outputs))

    # model_file = os.path.join(data_root, model_name)
    # if load_model and os.path.isfile(model_file + ".index"):
        # model.load_weights(model_file)
        # print("Loading model weights from ", model_file)
    # model.compile(
        # optimizer=tf.train.AdamOptimizer(),
        # loss='mean_squared_error'
    # )

    # # Training params
    # batch_size = 10
    # epochs = 30
    # # Fit the data
    # history = model.fit(dataset.batch(batch_size).repeat(epochs),
              # epochs=epochs, steps_per_epoch=len(list(dataset))//batch_size)

    # losses = history.history['loss']
    # plt.plot(losses)
    # plt.show()

    # model.save_weights(model_file)
    # print("Saving model weights to ", model_file)

    # Data Gen testing
    params = {'img_size': (120, 120),
              'channels': 'rgb',
              'batch_size': 2,
              'shuffle': True}
    train_dir = '/home/mmmfarrell/DeepTurtleControl/data/overfit_1'
    training_generator = DataGenerator(train_dir, **params)
    # val_generator = DataGenerator(train_dir, **params)
    # X, Y = training_generator.__getitem__(0)


    # Resnet50 testing
    input_shape = (*params['img_size'], 3)
    # Build Model
    # Resnet model
    resnet_model = ResNet50(weights='imagenet', include_top=False,
            input_shape=input_shape)

    # Our FC layers
    x = layers.Flatten()(resnet_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    angle_out = layers.Dense(1, activation='linear')(x)

    # Final model
    transfer_model = Model(inputs=resnet_model.input, outputs=angle_out)

    # transfer_model = Sequential()
    # transfer_model.add(layers.Dense(52, activation='relu'))
    # transfer_model.add(layers.Dense(26, activation='relu'))
    # transfer_model.add(layers.Dense(1, activation='tanh'))


    # Load image and preprocess
    # img_path = '/home/mmmfarrell/elephant.jpg'
    # img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # preprocessed_img = preprocess_input(x)

    # Compile model and train
    transfer_model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='mean_squared_error')
    transfer_model.fit_generator(generator=training_generator,
                                 # validation_data=val_generator,
                                 epochs=500,
                                 use_multiprocessing=True,
                                 workers=4)

    # x_train = np.random.random((1, 120, 120, 3))
    # y_train = np.zeros((1, 1)) + 0.15

    # transfer_model.fit(x=x_train, y=y_train, batch_size=1, epochs=5)
    # transfer_model.fit(x=preprocessed_img, y=desired_output, batch_size=1,
            # epochs=500, steps)

    # Predict after training
    # preds = transfer_model.predict(preprocessed_img)
    preds = transfer_model.predict(x_train)
    print("y_train", y_train)
    print(preds.shape)
    print(preds)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    plot_model(transfer_model, to_file='transfer_model.png', show_shapes=True)
    # print(preds.shape)
    # input()


