import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import training
import tensorflow.keras as keras
import statistics

from src import model
from src import config
from src import util
from src import model_interface

MAX_EPOCHS = 10

class ModelPath(model.ModelBase):
    def __init__(self, uses_goal=False):
        """
        Args:
            model - model used for prediction
            uses_goal - bool, marks the model as using the goal position as input
        """
        self.uses_goal = uses_goal
        self.model = None


    def train(self, model, train_data, eval_data, train_goals=None, eval_goals=None):
        """
        Trains the model,
        """
        assert not self.uses_goal or (train_goals is not None and eval_goals is not None)
        self._init_model_before_training(model)

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


    def metrics(self, path_input, ground_truth, filepath=None, goals=None):
        """
        Calculates metrics for the entire dataset.
            x - input path
            ground_truth - actual path taken by the pedestrian
            filepath - if given will save as csv
            goals - pedestrian goal input, necessary for models that use the goal
        """
        x = path_input

        if self.uses_goal:
            if goals is None:
                print("No goals provided, although model depends on it!")
                return None
            
            # if the model requires goals, then we need to add the right input as well
            x = model_interface.concatenate_x_goal(path_input, goals)

        predictions = self.model(x)
        mde_distances = util.mean_euclidean_distances(ground_truth, predictions)
        fde_distances = util.final_displacement_error(ground_truth, predictions)

        if filepath:
            util.save_array_to_file(filepath + '_mde.csv', mde_distances)
            util.save_array_to_file(filepath + '_fde.csv', fde_distances)

        mde_mean = statistics.mean(mde_distances)
        fde_mean = statistics.mean(fde_distances)
        print(f"MED: {mde_mean}")
        print(f"FDE: {fde_mean}")