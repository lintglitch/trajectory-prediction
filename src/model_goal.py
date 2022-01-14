import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
import tensorflow.keras as keras
from keras import layers

from src import model
from src import config
from src import util
from src import model_interface

MAX_EPOCHS = 10

class ModelGoal(model.ModelBase):
    def __init__(self):
        """
        Inputs:
            train_data - for infering the correct input shape
        """
        self.model = None


    def train(self, model, train_data, eval_data):
        self._init_model_before_training(model)

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
        tensor = self.model(data[0])
        arr = tensor.numpy()
        return arr