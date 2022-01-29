import pandas as pd
import numpy as np
import random
import logging
from tqdm import tqdm

from src import config
from src import util


def generate_sequences(df):
    """
    Main parsing function. Takes a pandas file with the following relevant columns: id, x, y, time

    Returns a list of people. Has the following format: a list of lists of (x, goal, y)
        Every unique person has a list of (x, goal, y)
    """

    ids = df.id.unique()
    logging.info("%s unique ids", len(ids))
    logging.info("Loading...")
    
    # contains all people paths
    all_persons_list = []

    # skipped trajectories
    skipped_too_short = 0
    skipped_frequency_problem = 0

    # skipped sequences
    skipped_absolute_too_high = 0

    # total generated sequences
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
            # note that the data is always rotated, this is to make sure there are not preferrential paths
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

                # skip all trajectories with too large absolute values
                if max_val >= 1.0:
                    skipped_absolute_too_high += 1
                    continue

                assert max_val <= 1.1, "Maximum value is {}".format(max_val)

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
    
    print(f"{len(ids)} unique ids found. Generated {gen_total} sequences total from {len(all_persons_list)} trajectories.")
    skipped_trajectories_total = skipped_too_short + skipped_frequency_problem
    print(f"""Skipped {skipped_trajectories_total} trajectories.
        {skipped_too_short} too short
        {skipped_frequency_problem} frequency issues (like gaps)
    """)

    print(f"Skipped {skipped_absolute_too_high} sequences ({round((skipped_absolute_too_high * 100) / gen_total, 3)}%) because of abnormally high absolute value after normalization")
    
    return all_persons_list


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

    x = np.array(x_poses, dtype='float32')
    y = np.array(y_poses, dtype='float32')
    goals = np.array(goals_list, dtype='float32')
    
    return x, goals, y


def divide_and_format(persons, train_ratio=0.7, eval_ratio=0.2):
    """
    Divides human trajectories into train, eval and test.
    """
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


def parse_atc_day(file_name, train_ratio=0.8, eval_ratio=0.2):
    """
    Parses a single csv atc file, which always represents a day.
    
    Splits dataset into train, eval and test. Test set will only be generated if train and eval percentages do not sum to 1.0.

    Returns: (train_data, eval_data, test_data)

    Each of these is a tuple of the following format: (x, goals, y)
        x - [batches, INPUT_FRAME_NUMBER, 2]
        goals - [batches, GOAL_GRID_SIZE**2], its a one hot encoded vector
        y - [batches, OUTPUT_FRAME_NUMBER, 2]
    """
    file_path = f"data/{file_name}"

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

    persons = generate_sequences(df)

    # plt.hist(max_vals)
    # plt.show()

    # return train_data, eval_data, test_data
    data = divide_and_format(persons, train_ratio=train_ratio, eval_ratio=eval_ratio)
    return data


def parse_several_atc_days(file_name_list, train_ratio=0.8, eval_ratio=0.2):
    train_datas = []
    eval_datas = []
    test_datas = []
    for file_name in file_name_list:
        data = parse_atc_day(file_name, train_ratio=train_ratio, eval_ratio=eval_ratio)
        train_data, eval_data, test_data = data
        train_datas.append(train_data)
        eval_datas.append(eval_data)

        if test_data is not None:
            test_datas.append(test_data)
          

    train_data_combined = unite_data(train_datas)
    eval_data_combined = unite_data(eval_datas)
    test_data_combined = None

    print("\nCOMBINED STATS:")
    print("combined train", train_data_combined[0].shape, train_data_combined[0].dtype, train_data_combined[1].shape, train_data_combined[1].dtype, train_data_combined[2].shape, train_data_combined[2].dtype)
    print("combined eval", eval_data_combined[0].shape, eval_data_combined[0].dtype, eval_data_combined[1].shape, eval_data_combined[1].dtype, eval_data_combined[2].shape, eval_data_combined[2].dtype)

    if test_datas:
        test_data_combined = unite_data(test_datas)
        print("combined test", test_data_combined[0].shape, test_data_combined[0].dtype, test_data_combined[1].shape, test_data_combined[1].dtype, test_data_combined[2].shape, test_data_combined[2].dtype)

    return train_data_combined, eval_data_combined, test_data_combined


def save_processed_data(filename, train_data, eval_data, test_data=None):
    path = f"data_processed/{filename}"

    if test_data is not None:
        np.savez(path,
            train_data_0=train_data[0], train_data_1=train_data[1], train_data_2=train_data[2],
            eval_data_0=eval_data[0], eval_data_1=eval_data[1], eval_data_2=eval_data[2],
            test_data_0=test_data[0], test_data_1=test_data[1], test_data_2=test_data[2]
        )
    else:
        np.savez(path,
            train_data_0=train_data[0], train_data_1=train_data[1], train_data_2=train_data[2],
            eval_data_0=eval_data[0], eval_data_1=eval_data[1], eval_data_2=eval_data[2]
        )


def load_processed_data(filename):
    """
    returns (train_data, eval_data, test_data)
    """

    path = f"data_processed/{filename}.npz"
    npzfile = np.load(path)

    train_data = [npzfile['train_data_0'], npzfile['train_data_1'], npzfile['train_data_2']]
    eval_data = [npzfile['eval_data_0'], npzfile['eval_data_1'], npzfile['eval_data_2']]

    # if test_data was generated load it
    test_data = None
    if 'test_data_0' in npzfile:
        test_data = [npzfile['test_data_0'], npzfile['test_data_1'], npzfile['test_data_2']]

    return train_data, eval_data, test_data
    

def unite_data(data_list):
    """
    Takes list of (input_paths, goals, output_paths) tuples and combines them into a whole.
    """

    input_paths = []
    goals = []
    output_paths = []
    for data in data_list:
        input_paths.append(data[0])
        goals.append(data[1])
        output_paths.append(data[2])
    
    input_paths_combined = np.concatenate( input_paths )
    goals_combined = np.concatenate( goals )
    output_paths_combined = np.concatenate( output_paths )

    return input_paths_combined, goals_combined, output_paths_combined


def unite_processed_data(file_name_list):
    train_datas = []
    eval_datas = []
    test_datas = []
    for file_name in file_name_list:
        data = load_processed_data(file_name)
        train_data, eval_data, test_data = data
        train_datas.append(train_data)
        eval_datas.append(eval_data)

        if test_data is not None:
            test_datas.append(test_data)
          

    train_data_combined = unite_data(train_datas)
    eval_data_combined = unite_data(eval_datas)
    test_data_combined = None

    print("\nCOMBINED STATS:")
    print("combined train", train_data_combined[0].shape, train_data_combined[0].dtype, train_data_combined[1].shape, train_data_combined[1].dtype, train_data_combined[2].shape, train_data_combined[2].dtype)
    print("combined eval", eval_data_combined[0].shape, eval_data_combined[0].dtype, eval_data_combined[1].shape, eval_data_combined[1].dtype, eval_data_combined[2].shape, eval_data_combined[2].dtype)

    if test_datas:
        test_data_combined = unite_data(test_datas)
        print("combined test", test_data_combined[0].shape, test_data_combined[0].dtype, test_data_combined[1].shape, test_data_combined[1].dtype, test_data_combined[2].shape, test_data_combined[2].dtype)

    return train_data_combined, eval_data_combined, test_data_combined


# frame rate: 30
def parse_stanford(file_path):
    df = pd.read_csv("video0/annotations.txt", sep=' ', names=["id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"])
    df['x'] = (df['xmin'] + df['xmax']) // 2
    df['y'] = (df['ymin'] + df['ymax']) // 2