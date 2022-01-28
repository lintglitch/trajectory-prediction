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


class ModelGoal(model.ModelBase):
    def __init__(self):
        """
        Inputs:
            train_data - for infering the correct input shape
        """
        self.model = None

        super().__init__()


    def train(self, model, train_data, eval_data, batch_size=128, epochs=10):
        self._init_model_before_training(model)

        # use past path as input
        train_x = train_data[0]
        eval_x = eval_data[0]

        # expect the future goal as output
        train_y = train_data[1]
        eval_y = eval_data[1]

        self.model.compile(loss='categorical_crossentropy',
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.keras.metrics.CategoricalAccuracy()])
                    # metrics=[tf.metrics.MeanAbsoluteError()])

    #     hist = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epochs, shuffle=True,
    #                             verbose=True, validation_data=(x_val, y_val), callbacks=self.callbacks)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
        callbacks = [reduce_lr]

        history = self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=True,
                            shuffle=True, validation_data=(eval_x, eval_y),
                            callbacks=callbacks
                            )

        return history
    

    def estimate(self, data):
        tensor = self.model(data[0])
        arr = tensor.numpy()
        return arr