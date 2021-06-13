import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config
import util
import model_interface

MAX_EPOCHS = 20
CONV_WIDTH = 3
LSTM_DEPTH = 128


def compile_and_fit(model, train_x, train_y, eval_x, eval_y, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
                # metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
                        shuffle=True,
                        validation_data=(eval_x, eval_y),
                        callbacks=[early_stopping])
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
    model.add(layers.LSTM(LSTM_DEPTH))
    model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))

    ## train no-goal model
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