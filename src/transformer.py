# modified from: https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_classification_transformer.py
# based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from src import config



"""
## Introduction
This is the Transformer architecture from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
applied to timeseries instead of natural language.
This example requires TensorFlow 2.4 or higher.
## Load the dataset
We are going to use the same dataset and preprocessing as the
[TimeSeries Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)
example.
"""



"""
## Build the model
Our model processes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
timeseries.
"""



"""
We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.
The projection layers are implemented through `keras.layers.Conv1D`.
"""


# encoders
def attention_block(inputs, head_size, num_heads, ff_dim=None, dropout=0):
    if ff_dim is None:
        ff_dim = head_size

    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)

    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


"""
The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.
"""

# n_classes = len(np.unique(y_train))

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # stack attention blocks
    for _ in range(num_transformer_blocks):
        x = attention_block(x, head_size, num_heads, ff_dim, dropout)

    # TODO test if necessary
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    output_size = config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES

    # add a couple of heads
    for _ in range(mlp_units):
        x = layers.Dense(output_size, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    x = layers.Dense(output_size)(x)
    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)

    # outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# model = build_model(
#     input_shape,
#     head_size=256,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=4,
#     mlp_units=[128],
#     mlp_dropout=0.4,
#     dropout=0.25,
# )

# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#     metrics=["sparse_categorical_accuracy"],
# )

# callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

# model.fit(
#     x_train,
#     y_train,
#     validation_split=0.2,
#     epochs=200,
#     batch_size=64,
#     callbacks=callbacks,
# )