import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
import tensorflow.keras as keras
from keras import layers

from src import config

CONV_WIDTH = 3
LSTM_DEPTH = 128


def simple_lstm(train_data):
    ## simple LSTM (works both for goal or no goal)
    model = tf.keras.Sequential()
    model.add(layers.LSTM(LSTM_DEPTH))
    model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))
    return model


def simple_lstm_dropout(train_data):
    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.LSTM(LSTM_DEPTH)(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(rate=0.2)(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def simple_cnn(train_data):
    output_size = config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES

    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    # x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    # x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(output_size, activation='relu')(x)
    x = layers.Dense(output_size)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def simple_cnn_dropout(train_data):
    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(rate=0.2)(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

    # model = tf.keras.Sequential()
    # model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=2))
    # model.add(layers.Dropout(rate=0.1, training=True))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    # model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))


def simple_cnn_probability(train_data):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    # model.add(tfp.layers.DenseVariational(units=config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES, make_prior_fn=prior, make_posterior))
    # model.add(layers.Dense(tfp.layers.DenseFlipout(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES, activation='relu'))
    size = config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES
    model.add(tfp.layers.DenseVariational(size, make_prior_fn=prior, make_posterior_fn=posterior, activation="sigmoid"))
    # model.add(tfp.layers.DenseFlipout(size, activation="relu", name="dense_1"))
    # tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(
    # len(outputs)), activation=None, name="distribution_weights"),
    # tfp.layers.MultivariateNormalTriL(len(outputs), activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/n_batches), name="output")
    # model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))
    return model