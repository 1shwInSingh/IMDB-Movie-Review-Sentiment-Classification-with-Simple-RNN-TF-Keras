"""Lightweight NumPy inference engine for IMDB Sentiment Classifier.

Executes the Senior Regularized Bi-LSTM model with Dual Global Pooling
(Average + Max Pooling) and Layer Normalization in pure NumPy.
Runs in ~3ms without requiring TensorFlow in production (Vercel serverless friendly).
"""
import os
import re
import string
import numpy as np

VOCAB_SIZE = 10000
MAX_LEN = 200
PUNCT_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _run_lstm_seq(x_seq, W, U, b):
    """Keras-compatible LSTM returning the full sequence of hidden states."""
    T = len(x_seq)
    units = U.shape[0]
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    h_seq = np.zeros((T, units), dtype=np.float32)

    for t in range(T):
        xt = x_seq[t]
        z = np.dot(xt, W) + np.dot(h, U) + b
        zi, zf, zc, zo = np.split(z, 4)
        i = sigmoid(zi)
        f = sigmoid(zf)
        c_cand = np.tanh(zc)
        o = sigmoid(zo)
        c = f * c + i * c_cand
        h = o * np.tanh(c)
        h_seq[t] = h

    return h_seq


class NumpySentimentModel:
    """NumPy-powered Regularized Bidirectional LSTM with Dual Pooling."""

    def __init__(self, npz_path, vocab_list):
        self.vocab = vocab_list
        self.word2idx = {w: i + 2 for i, w in enumerate(vocab_list)}

        weights = np.load(npz_path)
        self.W_embed = weights["W_embed"]
        self.W_f = weights["W_f"]
        self.U_f = weights["U_f"]
        self.b_f = weights["b_f"]
        self.W_b = weights["W_b"]
        self.U_b = weights["U_b"]
        self.b_b = weights["b_b"]
        self.ln_gamma = weights["ln_gamma"]
        self.ln_beta = weights["ln_beta"]
        self.W_dense1 = weights["W_dense1"]
        self.b_dense1 = weights["b_dense1"]
        self.W_out = weights["W_out"]
        self.b_out = weights["b_out"]
        self._loaded = True

    def is_loaded(self):
        return self._loaded

    def predict(self, text: str) -> float:
        """Return probability of positive sentiment in [0.0, 1.0]."""
        cleaned = PUNCT_REGEX.sub("", text.lower())
        words = cleaned.split()
        toks = [self.word2idx.get(w, 1) for w in words][:MAX_LEN]

        if not toks:
            return 0.5

        if len(toks) < MAX_LEN:
            toks = toks + [0] * (MAX_LEN - len(toks))

        toks_arr = np.array(toks, dtype=np.int32)
        x_seq = self.W_embed[toks_arr]

        # Forward LSTM (0 -> T)
        H_f = _run_lstm_seq(x_seq, self.W_f, self.U_f, self.b_f)

        # Backward LSTM (T -> 0)
        H_b_rev = _run_lstm_seq(x_seq[::-1], self.W_b, self.U_b, self.b_b)
        H_b = H_b_rev[::-1]

        # Concatenate forward + backward states at each sequence position
        H = np.concatenate([H_f, H_b], axis=-1)

        # Dual Global Pooling: overall thematic mood + salient emotional spikes
        avg_pool = np.mean(H, axis=0)
        max_pool = np.max(H, axis=0)
        concat = np.concatenate([avg_pool, max_pool])

        # Layer Normalization (epsilon=1e-3)
        mean = np.mean(concat)
        var = np.var(concat)
        norm = (concat - mean) / np.sqrt(var + 1e-3)
        norm = norm * self.ln_gamma + self.ln_beta

        # Dense projection with ReLU
        h1 = np.maximum(0.0, np.dot(norm, self.W_dense1) + self.b_dense1)

        # Dense output with Sigmoid
        prob = sigmoid(np.dot(h1, self.W_out) + self.b_out)[0]
        return float(prob)


class TFSentimentModel:
    """Fallback TensorFlow model wrapper when TF is installed and weights.h5 is used."""

    def __init__(self, h5_path, vocab_list):
        from model_def import build_model
        import tensorflow as tf

        self.tf = tf
        self.model = build_model(vocab_list)
        self.model.load_weights(h5_path)
        self._loaded = True

    def is_loaded(self):
        return self._loaded

    def predict(self, text: str) -> float:
        pred = self.model.predict(self.tf.constant([text]), verbose=0)
        return float(pred[0][0])


def load_model(base_dir, vocab_list):
    """Load model with priority: lightweight NumPy weights (.npz) -> TensorFlow (.weights.h5)."""
    npz_path = os.path.join(base_dir, "model_weights.npz")
    h5_path = os.path.join(base_dir, "imdb_sentiment_bilstm.weights.h5")

    # 1. Prefer lightweight pure NumPy weights (Vercel & production friendly)
    if os.path.exists(npz_path):
        return NumpySentimentModel(npz_path, vocab_list)

    # 2. Fall back to TensorFlow weights if available
    if os.path.exists(h5_path):
        try:
            return TFSentimentModel(h5_path, vocab_list)
        except Exception:
            pass

    return None
