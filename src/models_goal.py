import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
import tensorflow.keras as keras
from keras import layers

from src import config


def simple_cnn(train_data):
    output_size = config.GOAL_GRID_SIZE**2

    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    # x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(output_size, activation='relu')(x)
    outputs = layers.Dense(output_size, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model