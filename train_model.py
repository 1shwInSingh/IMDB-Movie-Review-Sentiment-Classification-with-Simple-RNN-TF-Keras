"""Train the Bidirectional LSTM IMDB sentiment model (mirrors the notebook).

Saves weights + vocabulary JSON instead of a full .keras model, because the
TextVectorization layer cannot be serialized/reloaded reliably (Keras 3).
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from model_def import VOCAB_SIZE, MAX_LEN, SEED, build_model

np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv("IMDB Dataset.csv")
df["review"] = df["review"].str.replace("<br />", " ", regex=False)
df = df.groupby("sentiment").sample(n=10000, random_state=SEED).reset_index(drop=True)
df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1}).astype(int)

X = df["review"].to_numpy(dtype=object)
y = df["sentiment"].to_numpy()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, stratify=y_train_full, random_state=SEED
)
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

vectorize_layer = keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
)
vectorize_layer.adapt(X_train)

vocab = vectorize_layer.get_vocabulary()
clean_vocab = list(dict.fromkeys(vocab))
clean_vocab = [t for t in clean_vocab if t not in ("", "[UNK]")][: VOCAB_SIZE - 2]

model = build_model(clean_vocab)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=15, batch_size=128, callbacks=[early_stop],
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))

model.save_weights("imdb_sentiment_bilstm.weights.h5")

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(clean_vocab, f, ensure_ascii=False)

with open("metrics.json", "w") as f:
    json.dump({
        "test_loss": round(float(test_loss), 3),
        "test_accuracy": round(float(test_accuracy), 3),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "train_size": int(len(X_train)), "val_size": int(len(X_val)), "test_size": int(len(X_test)),
    }, f, indent=2)
print("Saved imdb_sentiment_bilstm.weights.h5, vocab.json and metrics.json")
