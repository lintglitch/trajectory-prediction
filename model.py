import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
import tensorflow.keras as keras
from keras import layers

import config
import util
import model_interface

MAX_EPOCHS = 20
CONV_WIDTH = 3
LSTM_DEPTH = 128



class Model:
    def __init__(self, model, uses_goal=False):
        self.model = model
        self.uses_goal = uses_goal
    
    def train(self):
        pass
    

    def compile_and_fit(self, model, train_x, train_y, eval_x, eval_y, patience=2):
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                                 patience=patience,
        #                                                 mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
                    # metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
                            shuffle=True,
                            validation_data=(eval_x, eval_y))

        # history = model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
        #                     shuffle=True,
        #                     validation_data=(eval_x, eval_y),
        #                     callbacks=[early_stopping])
        return history


def train(model, train_data, eval_data, use_goal=False):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(LSTM_DEPTH))
    model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))


    ## train goal model
    train_data_goal_x = model_interface.concatenate_x_goal(train_data[0], train_data[1])
    eval_data_goal_x = model_interface.concatenate_x_goal(eval_data[0], eval_data[1])
    history_model = compile_and_fit(model, train_data_goal_x, train_data[2], eval_data_goal_x, eval_data[2])


def compile_and_fit(model, train_x, train_y, eval_x, eval_y, patience=2):
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                 patience=patience,
    #                                                 mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
                # metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
                        shuffle=True,
                        validation_data=(eval_x, eval_y))

    # history = model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
    #                     shuffle=True,
    #                     validation_data=(eval_x, eval_y),
    #                     callbacks=[early_stopping])
    return history



def simple_lstm(train_data, eval_data):
    # model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    #     # Shape => [batch, 1, conv_units]
    #     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    #     # Shape => [batch, 1,  out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS*num_features,
    #                           kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])

    ## simple LSTM (no goal)
    model = tf.keras.Sequential()
    model.add(layers.GRU(LSTM_DEPTH))
    # model.add(layers.LSTM(LSTM_DEPTH))
    model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))

    ## train no-goal model
    history_model = compile_and_fit(model, train_data[0], train_data[2], eval_data[0], eval_data[2])
    return model, history_model


def simple_lstm_dropout(train_data, eval_data):

    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.LSTM(LSTM_DEPTH)(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(rate=0.2)(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    history_model = compile_and_fit(model, train_data[0], train_data[2], eval_data[0], eval_data[2])
    return model, history_model


def simple_cnn(train_data, eval_data):
    output_size = config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES

    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    # x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(output_size, activation='relu')(x)
    x = layers.Dense(output_size)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    history_model = compile_and_fit(model, train_data[0], train_data[2], eval_data[0], eval_data[2])
    return model, history_model


def simple_cnn_dropout(train_data, eval_data):
    inputs = keras.Input(shape=train_data[0].shape[1:])
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(rate=0.2)(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    # model = tf.keras.Sequential()
    # model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=2))
    # model.add(layers.Dropout(rate=0.1, training=True))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    # model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))

    history_model = compile_and_fit(model, train_data[0], train_data[2], eval_data[0], eval_data[2])
    return model, history_model


def simple_cnn_probability(train_data, eval_data):
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

    history_model = compile_and_fit(model, train_data[0], train_data[2], eval_data[0], eval_data[2])
    return model, history_model


def simple_lstm_goal(train_data, eval_data):
    ## simple LSTM (goal)
    model = tf.keras.Sequential()
    model.add(layers.LSTM(LSTM_DEPTH))
    model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))


    ## train goal model
    train_data_goal_x = model_interface.concatenate_x_goal(train_data[0], train_data[1])
    eval_data_goal_x = model_interface.concatenate_x_goal(eval_data[0], eval_data[1])
    history_model = compile_and_fit(model, train_data_goal_x, train_data[2], eval_data_goal_x, eval_data[2])
    return model, history_model