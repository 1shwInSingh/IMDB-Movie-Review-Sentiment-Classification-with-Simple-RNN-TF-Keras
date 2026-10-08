"""Shared model definition — mirrors the final Bidirectional LSTM in the notebook."""
import tensorflow as tf
from tensorflow import keras

VOCAB_SIZE = 10000
MAX_LEN = 200
SEED = 42


def build_model(vocab_list):
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
    )
    vectorize_layer.set_vocabulary(vocab_list)
    return keras.Sequential([
        keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True),
        keras.layers.SpatialDropout1D(0.3, seed=SEED),
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, seed=SEED)
        ),
        keras.layers.Dense(
            1, activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(0.001),
        ),
    ])
