import scipy
import scipy.spatial
import numpy as np
import math
import statistics

import config

# taken from
# https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate_array(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


# ADE/MDE
# TODO also implement FDE
def mean_euclidean_distances(ground_truth, prediction):
    # print(ground_truth.shape, prediction.shape, ground_truth.shape[0], prediction.shape[0])
    batches = prediction.shape[0]
    prediction_length = prediction.shape[1]
    errors = []
    for b in range(batches):
        for t in range(prediction_length):
            error = scipy.spatial.distance.euclidean(ground_truth[b,t], prediction[b,t])
            errors.append(error)
    return errors
    # return statistics.mean(errors)


def get_goal_index(position):
    """
    Gets goal grid coordinates for 5x5 goal.
    """
    x, y = position

    col = 0
    gx = abs(x) / config.GOAL_SIZE
    if gx > 0.6:
        col = 2
    elif gx > 0.2:
        col = 1
    else:
        col = 0
    
    # get the correct sign
    col = int(math.copysign(col, x)) + 2
    assert(0 <= col <= 4) 

    row = 0
    gy = abs(y) / config.GOAL_SIZE
    if gy > 0.6:
        row = 2
    elif gy > 0.2:
        row = 1
    else:
        row = 0
    
    # get the correct sign
    row = -(int(math.copysign(row, y)) - 2)
    assert(0 <= row <= 4)

    return row * 5 + col


def goal_index_to_xy_indices(goal_index):
    row = int(goal_index // 5)
    col = goal_index % 5
    return col, row


def apply_randomness_to_goal_prediction(goal):
    pass
