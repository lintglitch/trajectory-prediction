import scipy
import scipy.spatial
import numpy as np
import math
import random
import statistics

from src import config

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
def average_displacement_error(ground_truth, prediction):
    """
    Calculates ADE for a single input / prediction example.
    """
    length = ground_truth.shape[0]
    assert length == prediction.shape[0]

    errors = []
    for t in range(length):
        error = scipy.spatial.distance.euclidean(ground_truth[t], prediction[t])
        errors.append(error)
    
    ade = statistics.mean(errors)
    return ade


# FDE
def final_displacement_error(ground_truth, prediction):
    assert ground_truth.shape[0] == prediction.shape[0]
    return scipy.spatial.distance.euclidean(ground_truth[-1], prediction[-1])


# mADE
def minimum_average_displacement_error(ground_truth, prediction_distribution):
    """
    Calculates for every time step the minimum displacement error between the ground truth position
    and the closest of the prediction positions for a single example.
    """
    length = ground_truth.shape[0]

    errors = []
    for t in range(length):
        time_step_errors = []
        for prediction in prediction_distribution:
                error = scipy.spatial.distance.euclidean(ground_truth[t], prediction[t])
                time_step_errors.append(error)
        
        errors.append(min(time_step_errors))
        
    m_ade = statistics.mean(errors)
    return m_ade


# mFDE
def minimum_final_displacement_error(ground_truth, prediction_distribution):
    length = ground_truth.shape[0]

    errors = []
    t = -1
    for prediction in prediction_distribution:
            error = scipy.spatial.distance.euclidean(ground_truth[t], prediction[t])
            errors.append(error)
           
    m_fde = min(errors)
    return m_fde


def epistemic_uncertainty_path(predictions):
    unified = np.array(predictions)
    output_time_steps = unified.shape[1]

    arr_total_var = []
    for time_step in range(output_time_steps):
        # calculate variances for x and y for every time step
        x_var = unified[:, time_step, 0].var()
        y_var = unified[:, time_step, 1].var()

        # since x and y are independent random variables, just sum them to get the total
        total_var = x_var + y_var
        arr_total_var.append(total_var)

    arr_total_std = np.sqrt(arr_total_var)
    return arr_total_std


def get_goal_index(position):
    """
    Gets goal grid coordinates for 5x5 goal from x, y positions.
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

    return row * config.GOAL_GRID_SIZE + col


def xy_indices_to_goal_index(x, y):
    """
    Returns goal index for given x, y indices of the goal grid. Will return
    None if outside of grid.
    """
    if not 0 <= x < config.GOAL_GRID_SIZE:
        return None
    
    if not 0 <= y < config.GOAL_GRID_SIZE:
        return None

    return x + y * config.GOAL_GRID_SIZE


def goal_index_to_xy_indices(goal_index):
    row = int(goal_index // config.GOAL_GRID_SIZE)
    col = goal_index % config.GOAL_GRID_SIZE
    return col, row


def apply_randomness_to_adjacent_goal_predictions(goal_predictions, amount=0.1, step_size=0.1, distance=1):
    """
    Makes absolute goal predictions less reliable, by given tiles adjacent to the goal
    a random chance. Will return a copy of the goal predictions with the randomness applied.

    Input:
        goal: linear numpy array of length Goal_Grid**2
        amount: amount of randomness, 1.0 means completely random goals
        step_size: in what step sizes the randomness should be added, smaller values result in more even distribution
        distance: how far away grid cells can be to be affected
    """
    # make copy
    goals_copy = np.copy(goal_predictions)
    
    # iterate over goals and randomize
    goal_amount = goals_copy.shape[0]
    for i in range(goal_amount):
        goal = goals_copy[i]
        assert(sum(goal) < 1.05)
        assert(0 < amount <= 1)

        # reduce goal by the randomness amount, so we keep total probability at 1.0
        goal *= 1.0 - amount

        randomness_added = 0
        goal_index = np.argmax(goal)
        goal_x, goal_y = goal_index_to_xy_indices(goal_index)

        while randomness_added < amount:
            x_dir = random.randint(-distance, distance)
            y_dir = random.randint(-distance, distance)

            x_index = goal_x + x_dir
            y_index = goal_y + y_dir
            new_goal_index = xy_indices_to_goal_index(x_index, y_index)

            # if the coordinates are outside bounds then try again
            if new_goal_index is None:
                continue

            goal[new_goal_index] += step_size
            randomness_added += step_size
        
        assert(sum(goal) >= 0.95)
        assert(sum(goal) < 1.0 + 2*step_size)

    return goals_copy


def save_array_to_file(filepath, array):
    with open(filepath, 'w') as f:
        for a in array:
            f.write(str(a) + '\n')