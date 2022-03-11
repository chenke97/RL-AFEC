import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# print(tf.__version__)
tf.get_logger().setLevel('ERROR')
# Given 90-d non-reference video data, output video quality score 0-100

def build_model():
    model = keras.Sequential([
    layers.Dense(90, activation='relu', input_shape=[90]),
#     layers.Dense(90, activation='relu', input_shape=[90]),
    layers.Dense(90, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
#     model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

def load_model():
    NN_model = build_model()
    # NN_model.summary()
    wights_file = 'Weights-494--5.61865.hdf5' # choose the best checkpoint 
    NN_model.load_weights(wights_file) # load it

    return NN_model


