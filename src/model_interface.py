from curses import raw
import numpy as np
import json
import re
import os

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


def save_history(filename, history):
    """
    Takes history object and saves history as a json.
    """
    path = f"histories/{filename}.json"

    with open(path, 'w') as f:
        json_s = json.dumps(history.history)
        f.write(json_s)


def load_history(filename, is_goal=False):
    """
    Loads history json or txt file.

    Inputs:
        filename - filename under histories folder
        is_goal - bool, should be true if goal predictor history
    """
    path_json = f"histories/{filename}.json"
    path_txt = f"histories/{filename}.txt"

    history_dict = None
    if os.path.exists(path_json):
        with open(path_json) as f:
            raw_text = f.read()
            history_dict = json.loads(raw_text)
    else:
        history_dict = _load_txt_history(path_txt, is_goal=is_goal)
    
    return history_dict


def _load_txt_history(path, is_goal=False):
    """
    Generates a history dict from the saved text output.

    Will assume its from path training unless goal set to true
    """

    with open(path) as f:
        raw_text = f.read()
    
    search_vals = None
    if is_goal:
        search_vals = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy']
    else:
        search_vals = ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
    
    history_dict = {}
    for key in search_vals:
        search_string = f" {key}: ([\d\.]*)"
        finds = re.findall(search_string, raw_text)
        finds_float = [float(x) for x in finds]

        history_dict[key] = finds_float
    
    return history_dict
