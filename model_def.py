"""Shared model definition — Senior AI Engineer Regularized Bi-LSTM with Dual Pooling."""
import tensorflow as tf
from tensorflow import keras

VOCAB_SIZE = 10000
MAX_LEN = 200
SEED = 42


def build_model(vocab_list):
    """Construct the cuDNN-accelerated regularized Bidirectional LSTM model with Dual Pooling."""
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
    )
    vectorize_layer.set_vocabulary(vocab_list)

    inputs = keras.Input(shape=(1,), dtype=tf.string, name="review_text")
    x = vectorize_layer(inputs)

    x = keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=64,
        embeddings_regularizer=keras.regularizers.l2(5e-5),
        name="embedding"
    )(x)
    x = keras.layers.SpatialDropout1D(0.4, seed=SEED, name="spatial_dropout")(x)

    # cuDNN compliant Bi-LSTM with full sequence return
    lstm_out = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=48,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.0, # Enables cuDNN GPU kernel acceleration
            seed=SEED
        ),
        name="bilstm"
    )(x)

    # Concatenated Pooling: global average (overall mood) + global max (key emotional signals)
    avg_pool = keras.layers.GlobalAveragePooling1D(name="avg_pool")(lstm_out)
    max_pool = keras.layers.GlobalMaxPooling1D(name="max_pool")(lstm_out)
    concat = keras.layers.Concatenate(name="pooling_concat")([avg_pool, max_pool])

    norm = keras.layers.LayerNormalization(name="layer_norm")(concat)
    dense_proj = keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(2e-4),
        name="dense_proj"
    )(norm)
    drop = keras.layers.Dropout(0.45, seed=SEED, name="head_dropout")(dense_proj)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="sentiment_prob")(drop)

    return keras.Model(inputs=inputs, outputs=outputs, name="Senior_BiLSTM_Sentiment")
