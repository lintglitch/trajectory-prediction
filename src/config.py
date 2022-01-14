# number of input features without goal
NUM_INPUT_FEATURES = 2
# targeted time in seconds
SEQUENCE_TIME = 5.0
# minimum time of path to be used (in seconds)
MIN_SEQUENCE_TIME = 2 * SEQUENCE_TIME

# x,y in mm, 1000x is 1m, this should be roughly the largest distance a person might travel in the given timeframe
SCALING_FACTOR = 10000

# size of the goal relative to the scaling factor
# increase this above 1 to allow goals further away then the grid
# TODO does goal size make sense? grid does not support this as of yet, neither does the visualization
GOAL_SIZE = 1
# width and height of the goal grid in number of cells, e.g. 5 would be a 5 x 5 grid
GOAL_GRID_SIZE = 5

## Sequence slicing parameters
# number of frames each second
FRAME_FREQUENCY = 10
# number of input frames
# ms per frame * seconds
INPUT_FRAME_NUMBER = FRAME_FREQUENCY * 4
# number of output frames
OUTPUT_FRAME_NUMBER = FRAME_FREQUENCY * 4
# offset from y horizon that should be used as goal point
# TODO right now goal is always 3 seconds behind the horizon, make this random
GOAL_FRAME_OFFSET = FRAME_FREQUENCY * 3

## Data Augmentation parameters
# number of frames between window slices
WINDOW_SLICING_DISTANCE = 4
# amount of additional paths generated by rotation
ROTATION_AMOUNT = 5