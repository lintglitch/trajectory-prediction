import tensorflow.keras as keras

class ModelBase:
    def save(self, filename):
        assert self.model is not None
        self.model.save( f"weights/{filename}" )
    

    def load(self, filename):
        self.model = keras.models.load_model( f"weights/{filename}" )
    

    def _init_model_before_training(self, model):
        assert self.model is None

        self.model = model