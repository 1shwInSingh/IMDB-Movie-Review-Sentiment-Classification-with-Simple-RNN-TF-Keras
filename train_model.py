"""Train the Senior Regularized Bidirectional LSTM IMDB sentiment model.

Features:
- cuDNN GPU compatible Bi-LSTM (recurrent_dropout=0)
- Dual Global Pooling (Average + Max)
- LayerNormalization + AdamW Decoupled Weight Decay + Label Smoothing
- Saves weights + vocabulary JSON + compact NumPy weights for Vercel/serverless.
"""
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from model_def import VOCAB_SIZE, MAX_LEN, SEED, build_model

np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU acceleration check
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[+] GPU Hardware Acceleration: DETECTED ({len(gpus)} GPU(s))")
else:
    print("[-] Running on optimized CPU with oneDNN (cuDNN-ready architecture).")

df = pd.read_csv("IMDB Dataset.csv")
df["review"] = df["review"].str.replace("<br />", " ", regex=False)
df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1}).astype(int)

# Use balanced 36,000 samples for superior generalization
SAMPLE_SIZE = 18000
df = df.groupby("sentiment").sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

X = df["review"].to_numpy(dtype=object)
y = df["sentiment"].to_numpy()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.12, stratify=y_train_full, random_state=SEED
)
print(f"[*] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

vectorize_layer = keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
)
vectorize_layer.adapt(X_train)

vocab = vectorize_layer.get_vocabulary()
clean_vocab = list(dict.fromkeys(vocab))
clean_vocab = [t for t in clean_vocab if t not in ("", "[UNK]")][: VOCAB_SIZE - 2]

model = build_model(clean_vocab)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.0008, weight_decay=0.001),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    ),
]

t0 = time.time()
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=10, batch_size=128, callbacks=callbacks,
)
train_time = round(time.time() - t0, 1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))

# 1. Save standard Keras weights
model.save_weights("imdb_sentiment_bilstm.weights.h5")

# 2. Save compact weights for serverless deployment (Vercel)
embed_layer = model.get_layer("embedding")
bilstm_layer = model.get_layer("bilstm")
layer_norm = model.get_layer("layer_norm")
dense_proj_layer = model.get_layer("dense_proj")
output_layer = model.get_layer("sentiment_prob")

np.savez_compressed(
    "model_weights.npz",
    W_embed=embed_layer.get_weights()[0],
    W_f=bilstm_layer.forward_layer.cell.get_weights()[0],
    U_f=bilstm_layer.forward_layer.cell.get_weights()[1],
    b_f=bilstm_layer.forward_layer.cell.get_weights()[2],
    W_b=bilstm_layer.backward_layer.cell.get_weights()[0],
    U_b=bilstm_layer.backward_layer.cell.get_weights()[1],
    b_b=bilstm_layer.backward_layer.cell.get_weights()[2],
    ln_gamma=layer_norm.get_weights()[0],
    ln_beta=layer_norm.get_weights()[1],
    W_dense1=dense_proj_layer.get_weights()[0],
    b_dense1=dense_proj_layer.get_weights()[1],
    W_out=output_layer.get_weights()[0],
    b_out=output_layer.get_weights()[1],
)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(clean_vocab, f, ensure_ascii=False)

with open("metrics.json", "w") as f:
    json.dump({
        "test_loss": round(float(test_loss), 3),
        "test_accuracy": round(float(test_accuracy), 3),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "train_size": int(len(X_train)), "val_size": int(len(X_val)), "test_size": int(len(X_test)),
        "train_time_seconds": train_time,
    }, f, indent=2)

print("Saved imdb_sentiment_bilstm.weights.h5, model_weights.npz, vocab.json and metrics.json")
