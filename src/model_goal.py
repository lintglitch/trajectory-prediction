import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
import tensorflow.keras as keras
from keras import layers
import math

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


    def train(self, train_data, eval_data, batch_size=128, epochs=10, checkpoints_path=None, model=None, learning_rate=0.001, learning_rate_scheduler=False):
        self._init_model_before_training(model, checkpoints_path)

        # use past path as input
        train_x = train_data[0]
        eval_x = eval_data[0]

        # expect the future goal as output
        train_y = train_data[1]
        eval_y = eval_data[1]


        if not self.loaded_checkpoint:
            self.model.compile(loss='categorical_crossentropy',
                        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        metrics=[tf.keras.metrics.CategoricalAccuracy()])


        # configure checkpoint saving
        callbacks = None
        if checkpoints_path is not None:
            checkpoints = keras.callbacks.ModelCheckpoint(
                filepath=checkpoints_path,
                monitor='val_categorical_accuracy',
                save_best_only=True, 
                verbose=0,
                mode='max'
                )
            callbacks = [checkpoints]

        # adds the learning rate scheduler with warm up
        if learning_rate_scheduler:
            callbacks += [keras.callbacks.LearningRateScheduler(lr_fast_slow, verbose=1)]

        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
        #                                               min_lr=0.0001)
        # callbacks = [reduce_lr]

        history = self.model.fit(
            x=train_x,
            y=train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=True,
            shuffle=True,
            validation_data=(eval_x, eval_y),
            callbacks=callbacks
        )

        return history
    

    def estimate(self, data, batch_limit=5000):
        """
        Makes an estimation call. The batch limit is the maximum amount of batches the network may receive at once.
        """

        input_data = data[0]
        input_batches = input_data.shape[0]

        runs = math.ceil(input_batches / batch_limit)

        output_arrs = []

        lower_index = 0
        higher_index = batch_limit
        for i in range(runs):
            print(f"{i+1}/{runs}: {lower_index}:{higher_index}")

            # call model
            input_run = input_data[lower_index:higher_index]
            tensor = self.model(input_run)
            arr = tensor.numpy()
            output_arrs.append(arr)

            # update indices
            lower_index += batch_limit
            higher_index = min(input_batches, higher_index+batch_limit)

        return np.concatenate(output_arrs)
    

    def evaluate(self, test_data, batch_size=256):
        input_data = test_data[0]
        correct_data = test_data[1]
        result = self.model.evaluate(input_data, correct_data, batch_size=batch_size)
        return result


def lr_fast_slow(epoch, lr, warmup_epochs=15, base_lr=1e-4, fast_lr=1e-3):
    # start slow for warm up
    if epoch <= warmup_epochs:
        return fast_lr
    
    return base_lr