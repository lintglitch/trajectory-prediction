from statistics import mode
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src import util
from src import model_interface


def draw_complete_path(frame, name):
    """
    Draws the complete trajectory of a single person.
    """
    x_vals = frame.x.to_list()
    y_vals = frame.y.to_list()
    
    plt.title(name)
    plt.plot(x_vals, y_vals)
    plt.show()


def draw_input_path(x, y, goal_pos, goal_index):
    """
    Draws input data formatted to size.
    """
    xvals_of_x = x[:,0]
    yvals_of_x = x[:,1]
    xvals_of_y = y[:,0]
    yvals_of_y = y[:,1]

    current_x = xvals_of_x[-1]
    current_y = yvals_of_x[-1]

    fig, ax = plt.subplots()
    fig.set_size_inches(8.0, 8.0)
    ax.set_xticks([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    ax.set_yticks([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # draw the goal pos
    goal_index_x, goal_index_y = util.goal_index_to_xy_indices(goal_index)
    scaling_factor = 0.4
    print(goal_index_x, goal_index_y)
    rect = patches.Rectangle((goal_index_x*scaling_factor - 1, -goal_index_y*scaling_factor + 0.6), 0.4, 0.4, facecolor='lightgray')
    ax.add_patch(rect)

    # draw past path
    ax.plot(xvals_of_x, yvals_of_x, 'b')
    # draw future path
    ax.plot(xvals_of_y, yvals_of_y, 'r')
    # draw current position point
    ax.plot(current_x, current_y, 'bo')
    # draw goal point
    ax.plot(goal_pos[0], goal_pos[1], 'ro')

    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.show()


def draw_path(x, ground_truth, predictions=None, goal=None, name=None, min_goal_percent=0.05, legend=True):
    """
    Draws generated prediction.

    Arguments
        x: single batch input
        ground_truth: correct future path
        predictions: list of generated predictions
        goal: prediction of the pedestrian goal
        name: title of the output plot
        min_goal_percent: min goal percentage that still gets displayed
        legend: should a legend be created

    Output
        blue: input
        red: ground-truth
        green: prediction
    """
    # seperate the x/y coordinates of input and ground_truth
    x_xvals = x[:,0]
    x_yvals = x[:,1]
    gt_xvals = ground_truth[:,0]
    gt_yvals = ground_truth[:,1]

    # position robot is currently in
    current_x = x_xvals[-1]
    current_y = x_yvals[-1]

    # create the figure
    fig, ax = plt.subplots()
    fig.set_size_inches(8.0, 8.0)

    # if goal is given then draw that
    if goal is not None:
        # get indices in descending order of value
        indices = np.flip(goal.argsort())

        scaling_factor = 0.4
        # iterate through the indices, stop when the values become too low
        for i in indices:
            val = goal[i]
            assert val <= 1.0 and val >= 0

            if val <= min_goal_percent:
                break
            
            goal_index_x, goal_index_y = util.goal_index_to_xy_indices(i)
            x_pos = goal_index_x*scaling_factor - 1
            y_pos = -goal_index_y*scaling_factor + 0.6
            color_intensity = 1.0 - val / 2
            color = (color_intensity, color_intensity, color_intensity)
            rect = patches.Rectangle((x_pos, y_pos), scaling_factor, scaling_factor, facecolor=color)

            print(f"Goal ({goal_index_x}, {goal_index_y}): {val:.2f}")
            ax.add_patch(rect)

    # generate the goal grid
    ax.set_xticks([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    ax.set_yticks([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # draw past path
    past_path, = ax.plot(x_xvals, x_yvals, 'b', label='Past')

    # given the prediction model generate a prediction and draw it
    prediction_path = None
    if predictions is not None:
        for prediction in predictions:
            # prediction = model_interface.predict_once(prediction_model, x, goal=goal)
            # TODO, slightly change color
            px = prediction[:,0]
            py = prediction[:,1]
            prediction_path, = ax.plot(px, py, 'r', label='Prediction')

    # mark current position
    ax.plot(current_x, current_y, 'bo', label='Current position')

    # draw ground-truth path
    gt_path, = ax.plot(gt_xvals, gt_yvals, 'g', label='Ground-truth')

    if name is not None:
        plt.title(name)

    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])

    if legend:
        legend_handles = [past_path, gt_path]
        if prediction_path is not None:
            legend_handles.append(prediction_path)

        ax.legend(handles=legend_handles)

    plt.show()


def draw_path_batch(x, ground_truth, goals=None, prediction_model=None, n=1, skip=0, rnd=False, samples=1, name="path"):
    """
    Draws multiple predictions.
    Arguments:
        x - (batch, input-time-steps, features) matrix of positions
        ground_truth - (batch, output-time-steps, features) matrix
        goals - (batch, one-hot-vector)
        prediction_model - model to use for predictions
        n - number of plots to generate
        skip - skip this number of batches, makes no sense together with rnd
        rnd - wether the batch should be chosen randomly
        samples - number of prediction that should be made, should be 1 for deterministic models
        name - what to name the plots
    """
    assert(x.shape[0] == ground_truth.shape[0])

    if skip > 0 and rnd:
        print("WARNING, USED BOTH RANDOM SELECTION AND SKIP FOR DRAWING. IGNORING SKIP")
        skip = 0

    for i in range(n):
        batch_index = i
        if rnd:
            batch_index = random.randrange(0, x.shape[0])
        elif skip > 0:
            batch_index += skip
        
        assert batch_index < x.shape[0], f"Tried to access batch index {batch_index} of {x.shape[0]}"
        
        x_batch = x[batch_index]
        gt_batch = ground_truth[batch_index]
        goal_batch = None

        if goals is not None:
            goal_batch = goals[batch_index]

        plot_name = "{} {}".format(name, batch_index)

        predictions = None
        if prediction_model:
            if samples <= 1:
                predictions = [prediction_model.predict_once(x_batch, goal=goal_batch)]
            else:
                predictions = prediction_model.prediction_sampling(x_batch, samples=samples, goal=goal_batch)

        draw_path(x_batch, gt_batch, goal=goal_batch, predictions=predictions, name=plot_name)


def draw_histories(history_key, y_label, history_dicts, names, graph_name=None, save_file=None, max_x=None):
    """
    Draws the loss of one or several history files.

    Arguments:
        history_key - key for the loss function
        y_label - name of the y axis label
        history_dicts - list of history dicts
        names - the names for the graph
        graph_name - name of the graph
        save - if string will save figure under that name
    """
    fig, ax = plt.subplots()

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)



    for history_dict, name in zip(history_dicts, names):
        if history_key not in history_dict:
            print(f"{history_key} not in {name}")
            print(history_dict)
            continue

        data = history_dict[history_key]

        # so we start from epoch one add one more data point at the start
        data.insert(0, data[0])

        ax.plot(data, label=name)

    if graph_name:
        plt.title(graph_name)
    
    plt.xlim(left=1)
    if max_x:
        plt.xlim(right=max_x)
        
    plt.xlabel('epoch')
    plt.ylabel(y_label)

    ax.legend()

    if save_file is not None:
        plt.savefig(f"figures/{save_file}.pdf")

    plt.show()