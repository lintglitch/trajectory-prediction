import numpy as np
import statistics

from src import config
from src import util


def test_model(model, test_data):
    """
    Evaluates model with test data.
    """
    x = test_data[0]
    y = test_data[2]
    loss = model.evaluate(x=x, y=y, verbose=True)
    print("Test loss is {}".format(loss))


def concatenate_x_goal(x, goal):
    """
    Cobines x and goal into a (batch, time, 2 + goal_cells) matrix.

    Inputs:
        x - (batch, time, 2)
        goal - (time, 2)
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


# def predict_batch(model, x, goal=None):