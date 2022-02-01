# modified from: https://www.kaggle.com/yamqwe/tutorial-time-series-transformer-time2vec/notebook
# based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from src import config

class TransformerBlock(layers.Layer):
    def __init__(self, head_size, feat_dim, num_heads, ff_dim, rate = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads = num_heads, key_dim = head_size)
        self.ffn = keras.Sequential( [layers.Dense(ff_dim, activation = "gelu"), layers.Dense(feat_dim),] )
        self.layernorm1 = layers.BatchNormalization()
        self.layernorm2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)



class Time2Vec(layers.Layer):
    def __init__(self, kernel_size = 1):
        super(Time2Vec, self).__init__(trainable = True, name = 'Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name = 'wb', shape = (input_shape[1],), initializer = 'uniform', trainable = True)
        self.bb = self.add_weight(name = 'bb', shape = (input_shape[1],), initializer = 'uniform', trainable = True)
        # periodic
        self.wa = self.add_weight(name = 'wa', shape = (1, input_shape[1], self.k), initializer = 'uniform', trainable = True)
        self.ba = self.add_weight(name = 'ba', shape = (1, input_shape[1], self.k), initializer = 'uniform', trainable = True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)
        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))


# constant to muliply the skip connections
SKIP_CONNECTION_STRENGTH = 0.9

# N_HEADS = 8; num_heads
# FF_DIM = 256
# N_BLOCKS = 6; num_transformer_blocks
# EMBED_DIM = 64; head_size
# TIME_2_VEC_DIM = 3; time2vec_dim
# N_BLOCKS = 6; num_transformer_blocks



def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, time2vec_dim = 3):
    inp = layers.Input(input_shape)
    x = inp

    time_embedding = keras.layers.TimeDistributed(Time2Vec(time2vec_dim - 1))(x)
    x = layers.Concatenate(axis = -1)([x, time_embedding])
    x = layers.LayerNormalization(epsilon = 1e-6)(x)

    for k in range(num_transformer_blocks):
        x_old = x
        transformer_block = TransformerBlock(head_size, input_shape[-1] + ( input_shape[-1] * time2vec_dim), num_heads, ff_dim, dropout)
        x = transformer_block(x)
        x = ((1.0 - SKIP_CONNECTION_STRENGTH) * x) + (SKIP_CONNECTION_STRENGTH * x_old)

    x = layers.Flatten()(x)

    output_size = config.OUTPUT_FRAME_NUMBER*config.NUM_INPUT_FEATURES
    # add a couple of heads
    for _ in range(mlp_units):
        x = layers.Dense(8 * output_size, activation="selu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    x = layers.Dense(output_size, activation = 'linear')(x)

    outputs = layers.Reshape([config.OUTPUT_FRAME_NUMBER, config.NUM_INPUT_FEATURES])(x)
    model = keras.Model(inp, outputs)
    return model