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

MAX_EPOCHS = 10
CONV_WIDTH = 3
LSTM_DEPTH = 128



class Model:
    def __init__(self, model, uses_goal=False):
        """
        Args:
            model - model used for prediction
            uses_goal - bool, marks the model as using the goal position as input
        """
        self.model = model
        self.uses_goal = uses_goal

        print(self.model.summary())


    def train(self, train_data, eval_data, train_goals=None, eval_goals=None):
        """
        Trains the model,
        """
        assert self.model is not None
        assert not self.uses_goal or (train_goals and eval_goals)

        # x is just the past path, without goal
        train_x = train_data[0]
        eval_x = eval_data[0]

        # y is the future path taken
        train_y = train_data[2]
        eval_y = eval_data[2]

        # handle the goal estimation
        if self.uses_goal:
            # combine the goal estimation with the path input for the network
            train_x = model_interface.concatenate_x_goal(train_x, train_goals)
            eval_x = model_interface.concatenate_x_goal(eval_x, eval_goals)


        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                                 patience=patience,
        #                                                 mode='min')

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
                    # metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
                            shuffle=True,
                            validation_data=(eval_x, eval_y))

        # history = model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
        #                     shuffle=True,
        #                     validation_data=(eval_x, eval_y),
        #                     callbacks=[early_stopping])
        return history


    def predict_once(self, x_input, goal=None):
        """
        Makes single prediction using the model. Expects no batch dimension (batch size of one).

        Inputs:
            model - trained model
            x_input - input values of shape (time-steps, features)
                if model requires goal then x should either already be concatenated or goal must be given
            goal - will concatenate the goal to the x values before predicting
        
        Returns (time-steps, 2) matrix, consisting of the x/y positions.
        """
        if x_input is None:
            return None

        # combine the goal information into the input positions
        if goal is not None:
            x_input = model_interface.concatenate_x_goal_batch(x_input, goal)

        # add batch dimension
        x_input = np.expand_dims(x_input, axis=0)
        
        prediction = self.model(x_input)[0]
        return prediction


    def prediction_sampling(self, x, samples=100, goal=None):
        """
        Makes single prediction multiple times. Useful for non-deterministic models.
        """

        predictions = []
        for _ in range(samples):
            predictions.append(self.predict_once(x, goal=goal))
        
        return predictions
    

def simple_lstm(train_data):
    ## simple LSTM (no goal)
    model = tf.keras.Sequential()
    model.add(layers.GRU(LSTM_DEPTH))
    # model.add(layers.LSTM(LSTM_DEPTH))
    model.add(layers.Dense(config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES))
    model.add(layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES]))
    return model


def simple_lstm_goal(train_data):
    ## simple LSTM (goal)
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