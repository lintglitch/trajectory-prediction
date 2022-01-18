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


class ModelPath(model.ModelBase):
    def __init__(self, uses_goal=False):
        """
        Args:
            model - model used for prediction
            uses_goal - bool, marks the model as using the goal position as input
        """
        self.uses_goal = uses_goal
        self.model = None


    def train(self, model, train_data, eval_data, train_goals=None, eval_goals=None, batch_size=128, epochs=10):
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

        history = self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=True,
                            shuffle=True, validation_data=(eval_x, eval_y)
                            )

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

        # TODO optimize, instead of concatenating goal multiple times

        predictions = []
        for _ in range(samples):
            predictions.append(self.predict_once(x, goal=goal))
        
        # epistemic uncertainty
        print(predictions)

        # TODO: uncertainty
        # epistemic is just std, but how handle multiple?

        # you add together the variances, then take root again to get the combined std

        return predictions
    

    def prediction_sampling_and_uncertainty(self, x, samples=100, goal=None):
        pass


    def metrics_geometric(self, x, ground_truth, filepath=None, goals=None):
        """
        Calculates metrics for the entire dataset.
            x - input path
            ground_truth - actual path taken by the pedestrian
            filepath - if given will save as csv
            goals - pedestrian goal input, necessary for models that use the goal
        """
        input_x = x
        # extend input if goals are used
        if self.uses_goal:
            input_x = self._setup_goal_input(x, goals)

        predictions = self.model(input_x)
        mde_distances = util.average_displacement_error(ground_truth, predictions)
        fde_distances = util.final_displacement_error(ground_truth, predictions)

        if filepath:
            util.save_array_to_file(filepath + '_ADE.csv', mde_distances)
            util.save_array_to_file(filepath + '_FDE.csv', fde_distances)

        mde_mean = statistics.mean(mde_distances)
        fde_mean = statistics.mean(fde_distances)
        print(f"ADE mean: {mde_mean}")
        print(f"FDE mean: {fde_mean}")
    

    def metrics_probabilistic(self, x, ground_truth, samples=100, filepath=None, goals=None):
        """
        Calculates metrics for the entire dataset.
            x - input path
            ground_truth - actual path taken by the pedestrian
            filepath - if given will save as csv
            goals - pedestrian goal input, necessary for models that use the goal
        """
        input_x = x
        # extend input if goals are used
        if self.uses_goal:
            input_x = self._setup_goal_input(x, goals)

        predictions = []
        for _ in range(samples):
            predictions.append(self.model(input_x))

        mde_distances = util.minimum_average_displacement_error(ground_truth, predictions)
        fde_distances = util.minimum_final_displacement_error(ground_truth, predictions)

        if filepath:
            util.save_array_to_file(filepath + '_mADE.csv', mde_distances)
            util.save_array_to_file(filepath + '_mFDE.csv', fde_distances)

        mde_mean = statistics.mean(mde_distances)
        fde_mean = statistics.mean(fde_distances)
        print(f"mADE mean: {mde_mean}")
        print(f"mFDE mean: {fde_mean}")
    

    def _setup_goal_input(self, x, goals):
        if goals is None:
            print("No goals provided, although model depends on it!")
            return None
        
        # if the model requires goals, then we need to add the right input as well
        extended_x = model_interface.concatenate_x_goal(x, goals)

        return extended_x
