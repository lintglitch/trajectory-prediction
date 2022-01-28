import tensorflow as tf
import tensorflow.keras as keras

class ModelBase:
    def __init__(self) -> None:
        self.loaded_checkpoint = False

    def save(self, filename):
        assert self.model is not None
        self.model.save( f"weights/{filename}" )
    

    def load(self, filename):
        self.model = keras.models.load_model( f"weights/{filename}" )
    

    def _init_model_before_training(self, model=None, checkpoints_path=None):
        assert self.model is None
        assert model is not None or checkpoints_path is not None

        if not tf.config.list_physical_devices('GPU'):
            print('Warning: No GPU found. Training without GPU acceleration.')



        if model is not None:
            print("Compiling new model")
            self.model = model

        # we are loading an existing model from checkpoints
        else:
            print(f"Loading model from checkpoint data in {checkpoints_path}")

            self.model = keras.models.load_model(checkpoints_path)
            self.loaded_checkpoint = True