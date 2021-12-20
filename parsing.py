import pandas as pd
import numpy as np
import random
import logging
from tqdm import tqdm

import config
import util


def generate_sequences(df):
    """
    Returns a list of people.
    Has the following format: a list of lists of (x, goal, y)
        Every unique person has a list of (x, goal, y)
    """

    ids = df.id.unique()
    logging.info("%s unique ids", len(ids))
    logging.info("Loading...")
    
    # contains all people paths
    all_persons_list = []

    # contains the maximum values
    max_vals = []

    skipped_too_short = 0
    skipped_frequency_problem = 0
    gen_total = 0

    # filtered_sequences = []
    for name_id in tqdm(ids):
        # skip low ids, they are not real persons
        if name_id <= 1:
            continue

        # get only the elements relevant to the tracked person
        id_frame = df[df["id"] == name_id]
        # can now change the index to time, now that we observe single person there should
        # no longer be a time conflict
        id_frame = id_frame.set_index("time")
        
        # get sequence time in seconds
        time_length = (id_frame.index.max() - id_frame.index.min()).seconds
        # filter out too short sequences
        if time_length < config.MIN_SEQUENCE_TIME:
            logging.debug("Skipped too short sequence %s: %ss", name_id, time_length)
            skipped_too_short += 1
            continue
        
        # resample the sequence with variable refresh rate to constant rate
        # 100ms = 10Hz
        id_frame = id_frame.resample("100ms", origin='start', kind='period').first()

        # if tracking data has time-gaps then resampling will generate nan values, skip these sequences
        if id_frame.isnull().values.any():
            logging.debug("Skipped sequence with time-gaps %s", name_id)
            skipped_frequency_problem += 1
            continue

        person_list = []

        # get number of resampled frames
        frame_num = id_frame.shape[0]

        # select only the necessary columns
        id_frame = id_frame.loc[:, ['x', 'y']]

        logging.debug("Person %s: %ss, %s frames", name_id, time_length, frame_num)
        # draw_complete_path(id_frame, "{}_complete".format(name_id))

        # use window slicing to go through path
        index = 0
        gen_by_window_slicing = 0
        while True:
            # get indices for getting the right data
            start_index = index
            stop_index = start_index + config.INPUT_FRAME_NUMBER + config.OUTPUT_FRAME_NUMBER + config.GOAL_FRAME_OFFSET

            # stop if you go out of bounds
            if stop_index >= frame_num:
                break

            # get the relevant values and make copies of them, so the original data doesn't change
            data = id_frame.iloc[start_index:stop_index].to_numpy(copy=True)

            # the last position of the input is the current position of the human
            # substract 1 since one is length and this is an index
            current_pose_index = config.INPUT_FRAME_NUMBER -1
            current_pos = data[current_pose_index].copy()

            # make the values relative to the position
            data -= current_pos
            assert(data[current_pose_index,0] == 0 and data[current_pose_index,1] == 0)

            # create number of paths equal to the rotation amount, data augmentation
            for _ in range(config.ROTATION_AMOUNT):
                rotation_angle = random.randrange(0, 359)

                # rotate the data array around the current position (0, 0)
                rotated_data = util.rotate_array(data.copy(), degrees=rotation_angle)

                # start index of the y position
                y_index = config.INPUT_FRAME_NUMBER
                # end of the y values
                end_index = y_index + config.OUTPUT_FRAME_NUMBER

                # (time_steps_input, 2) matrix
                x = rotated_data[:y_index]
                # (time_steps_output, 2) matrix
                y = rotated_data[y_index:end_index]
                # length 2 vector used to convert into one-hot
                goal_pos = rotated_data[-1]
                assert(x[-1,0] == 0 and x[-1,1] == 0)

                # get the highest absolute value within that path
                max_val = max(abs(max(x.min(), x.max(), key=abs)), abs(max(y.min(), y.max(), key=abs)))
                max_vals.append(max_val)
                assert max_val <= 2.0, "Maximum value is {}".format(max_val)

                goal_index = util.get_goal_index(goal_pos)
                # draw_input_path(x, y, goal_pos, goal_index)    

                # goal index for 5x5
                goal_one_hot = np.zeros(25, dtype=np.float32)
                goal_one_hot[goal_index] = 1.0

                # save generated x, goal, y
                person_list.append( (x, goal_one_hot, y) )
                gen_total += 1

            index += config.WINDOW_SLICING_DISTANCE
            gen_by_window_slicing += 1
        
        if person_list:
            all_persons_list.append(person_list)
            logging.debug("Generated %s sequences multiplied by factor %s from slicing", len(person_list), gen_by_window_slicing)
        else:
            logging.debug("Empty person list generated")
    
    logging.info("Generated %s sequences total from %s trajectories.", gen_total, len(all_persons_list))
    logging.info("Skipped %s trajectories. %s too short. %s frequency issue (like gaps).",
        skipped_too_short + skipped_frequency_problem, skipped_too_short, skipped_frequency_problem)
    
    return all_persons_list, max_vals


def _format_x_y(persons):
    x_poses = []
    goals_list = []
    y_poses = []

    # go through all the people
    for person in persons:
        person_x_poses = [i[0] for i in person]
        person_goals   = [i[1] for i in person]
        person_y_poses = [i[2] for i in person]

        x_poses.extend(person_x_poses)
        goals_list.extend(person_goals)
        y_poses.extend(person_y_poses)

    x = np.array(x_poses)
    y = np.array(y_poses)
    goals = np.array(goals_list)
    
    return x, goals, y


def divide_and_format(persons, train_ratio=0.7, eval_ratio=0.2):
    assert(train_ratio + eval_ratio <= 1.0)

    # divide human paths into train, eval and test
    # by dividing them based on humans it is guaranteed that there is no
    # information leak

    # shuffle the paths
    random.shuffle(persons)

    # calculate indices
    num_person = len(persons)
    train_num = int(num_person * train_ratio)
    eval_num = 0
    test_num = 0
    if train_ratio + eval_ratio < 1:
        eval_num = int(num_person * eval_ratio)
        test_num = num_person - (train_num + eval_num)
    else:
        eval_num = num_person - train_num
    
    assert test_num >= 0

    print("Dividing {} human trajectories: {} training, {} eval, {} test".format(num_person, train_num, eval_num, test_num))

    train_data = persons[:train_num]
    eval_data = persons[train_num:train_num+eval_num]
    test_data = None
    if test_num > 0:
        test_data = persons[train_num+eval_num:]
    
    assert(train_num == len(train_data))
    assert(eval_num == len(eval_data))
    assert(test_data is None or test_num == len(test_data))

    train_data = _format_x_y(train_data)
    print("train", train_data[0].shape, train_data[0].dtype, train_data[1].shape, train_data[1].dtype, train_data[2].shape, train_data[2].dtype)
    eval_data = _format_x_y(eval_data)
    print("eval", eval_data[0].shape, eval_data[0].dtype, eval_data[1].shape, eval_data[1].dtype, eval_data[2].shape, eval_data[2].dtype)

    # load test data if available
    if test_data is not None:
        test_data = _format_x_y(test_data)
        print("test", test_data[0].shape, test_data[0].dtype, test_data[1].shape, test_data[1].dtype, test_data[2].shape, test_data[2].dtype)
    
    return train_data, eval_data, test_data


def parse_atc_day(file_path, train_ratio=0.8, eval_ratio=0.2):
    df = pd.read_csv(file_path, names=["time", "id", "x", "y", "z", "velocity", "motion_angle", "facing_angle"])
    # print(df.head())

    # remove unnecessary columns to save memory
    df = df.drop(columns=['z', 'velocity', 'motion_angle', 'facing_angle'])
    # print(df.dtypes)
    # print(df.head())

    # convert time
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # print(df.time.head())

    # scale x, y
    df['x'] = df['x'] / config.SCALING_FACTOR
    df['y'] = df['y'] / config.SCALING_FACTOR
    # print(df.dtypes)

    persons, max_vals = generate_sequences(df)

    # plt.hist(max_vals)
    # plt.show()

    # return train_data, eval_data, test_data
    data = divide_and_format(persons, train_ratio=train_ratio, eval_ratio=eval_ratio)
    return data