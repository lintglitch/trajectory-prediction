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

class ModelGoal:
    def __init__(self, model):
        """
        Inputs:
            train_data - for infering the correct input shape
        """
        self.model = model
        print(self.model.summary())


    def train(self, train_data, eval_data):
        assert self.model is not None

        # use past path as input
        train_x = train_data[0]
        eval_x = eval_data[0]

        # expect the future goal as output
        train_y = train_data[1]
        eval_y = eval_data[1]

        self.model.compile(loss='categorical_crossentropy',
                    optimizer=tf.optimizers.Adam(),
                    metrics=['accuracy'])
                    # metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(x=train_x, y=train_y, epochs=MAX_EPOCHS,
                            shuffle=True,
                            validation_data=(eval_x, eval_y))

        return history
    

    def estimate(self, data):
        return self.model(data[0])


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