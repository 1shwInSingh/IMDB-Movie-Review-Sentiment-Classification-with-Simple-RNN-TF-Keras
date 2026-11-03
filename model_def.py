"""Shared model definition — Senior AI Engineer High-Capacity Bi-LSTM with Dual Pooling.

Architecture rationale (v2 — underfitting fix):
  - LSTM units 48→96: doubles representational capacity for complex sentiment patterns
  - SpatialDropout 0.4→0.25: was over-constraining embedding space
  - head_dropout 0.45→0.30: allows classifier to learn a tighter decision boundary
  - Dense projection 32→64: wider head to handle richer 96*2=192-dim pooled features
  - L2 on embedding reduced 5e-5→2e-5: embedding weights were under-fitting word relationships
  - L2 on dense reduced 2e-4→5e-5: stop penalizing the head so heavily
  All changes cuDNN-compliant (recurrent_dropout=0.0 preserved for GPU acceleration).
"""
import tensorflow as tf
from tensorflow import keras

VOCAB_SIZE = 10000
MAX_LEN = 200
SEED = 42


def build_model(vocab_list):
    """Construct the cuDNN-accelerated high-capacity Bidirectional LSTM with Dual Pooling."""
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
    )
    vectorize_layer.set_vocabulary(vocab_list)

    inputs = keras.Input(shape=(1,), dtype=tf.string, name="review_text")
    x = vectorize_layer(inputs)

    x = keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=64,
        embeddings_regularizer=keras.regularizers.l2(2e-5),  # 5e-5→2e-5: less embedding penalty
        name="embedding"
    )(x)
    x = keras.layers.SpatialDropout1D(0.25, seed=SEED, name="spatial_dropout")  (x)  # 0.4→0.25

    # cuDNN-compliant Bi-LSTM — recurrent_dropout=0.0 preserves GPU kernel path
    lstm_out = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=96,               # 48→96: core capacity increase
            return_sequences=True,
            dropout=0.25,           # 0.3→0.25: slightly easier learning
            recurrent_dropout=0.0,  # Enables cuDNN GPU kernel acceleration
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
        64,                                              # 32→64: matches wider 192-dim input
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(5e-5), # 2e-4→5e-5: stop over-penalizing head
        name="dense_proj"
    )(norm)
    drop = keras.layers.Dropout(0.30, seed=SEED, name="head_dropout")(dense_proj)  # 0.45→0.30
    outputs = keras.layers.Dense(1, activation="sigmoid", name="sentiment_prob")(drop)

    return keras.Model(inputs=inputs, outputs=outputs, name="Senior_BiLSTM_v2_Sentiment")
