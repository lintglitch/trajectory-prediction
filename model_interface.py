import numpy as np
import statistics

import config
import util


def test_model(model, test_data):
    """
    Evaluates model with test data.
    """
    x = test_data[0]
    y = test_data[2]
    loss = model.evaluate(x=x, y=y, verbose=True)
    print("Test loss is {}".format(loss))


# TODO perhaps rewrite this to custom metrics class
def calculate_final_metrics(model, eval_x, eval_y, filepath=None):
    """
    Calculates metrics for the entire dataset.
        filepath - if given will save as csv
    """
    predictions = model(eval_x)
    distances = util.mean_euclidean_distances(eval_y, predictions)
    mean_distance = statistics.mean(distances)

    if filepath:
        with open(filepath, 'w') as f:
            for distance in distances:
                f.write(str(distance) + '\n')

    print(f"MED: {mean_distance}")


def concatenate_x_goal(x, goal):
    """
    Cobines x and goal into a (batch, time, 2 + goal_cells) matrix.
    """

    # repeat goal one-hot-vectors for each frame
    goal_ext = np.repeat(goal, config.INPUT_FRAME_NUMBER, axis=0).reshape(goal.shape[0], config.INPUT_FRAME_NUMBER, goal.shape[1])
    assert((goal[0] == goal_ext[0, 0]).all())
    assert((goal[0] == goal_ext[0, -1]).all())
    assert((goal[1] == goal_ext[1, 0]).all())
    assert((goal[1] == goal_ext[1, -1]).all())

    # combine both matrices together
    x_combined = np.concatenate( (x, goal_ext), axis=2)
    assert((x[0,0] == x_combined[0,0,0:2]).all())
    return x_combined


def concatenate_x_goal_batch(x, goal):
    """
    Cobines single x and goal batch into a (time, 2 + goal_cells) matrix.
    """
    assert x.shape == (config.INPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES)
    assert goal.shape == (25,)

    # repeat goal one-hot vector for length of input sequence
    goal_ext = np.expand_dims(goal, axis=0)
    goal_ext = np.repeat(goal_ext, x.shape[0], axis=0)

    x_combined = np.concatenate( (x, goal_ext), axis=1 )
    return x_combined


def predict_once(model, x, goal=None):
    """
    Makes single prediction using the model.

    Inputs:
        model - trained model
        x - input values of shape (time-steps, features)
            if model requires goal then x should either already be concatenated or goal must be given
        goal - will concatenate the goal to the x values before predicting
    
    Returns (time-steps, 2) matrix, consisting of the x/y positions.
    """
    if model is None or x is None:
        return None
    
    x_input = x

    # combine the goal information into the input positions
    if goal is not None:
        x_input = concatenate_x_goal_batch(x, goal)

    # add batch dimension
    x_input = np.expand_dims(x_input, axis=0)
    
    prediction = model(x_input)[0]
    return prediction