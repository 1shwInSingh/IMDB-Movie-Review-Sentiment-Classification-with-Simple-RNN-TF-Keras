"""Flask web frontend for the IMDB Bidirectional LSTM sentiment classifier."""
import json
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from model_def import build_model

app = Flask(__name__)

# Robust absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "imdb_sentiment_bilstm.weights.h5")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# Load vocabulary and build Bi-LSTM model
with open(VOCAB_PATH, encoding="utf-8") as f:
    vocab_list = json.load(f)

model = build_model(vocab_list)
model.load_weights(WEIGHTS_PATH)

# Warm up TensorFlow computation graph so initial user request has zero cold-start delay
try:
    _ = model.predict(tf.constant(["warmup query for graph compilation"]), verbose=0)
except Exception:
    pass

# Load evaluation metrics
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)


@app.route("/")
def index():
    """Render main interactive sentiment classification studio."""
    return render_template("index.html", metrics=metrics, metrics_json=json.dumps(metrics))


@app.route("/favicon.ico")
def favicon():
    """Prevent 404 logs when browsers request favicon."""
    return app.response_class("", status=204, mimetype="image/x-icon")


@app.route("/api/health")
def health():
    """Service health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "Bidirectional LSTM",
        "vocab_size": len(vocab_list),
        "weights_loaded": os.path.exists(WEIGHTS_PATH)
    })


@app.route("/api/metrics")
def get_metrics():
    """Return model training and evaluation metrics."""
    return jsonify(metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run neural sentiment inference on user submitted movie review."""
    start_t = time.perf_counter()
    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()

    if not review:
        return jsonify({"error": "Please enter a review."}), 400

    # Truncate overly long text safely to avoid unbounded compute
    if len(review) > 10000:
        review = review[:10000]

    prob = float(model.predict(tf.constant([review]), verbose=0)[0][0])
    latency_ms = round((time.perf_counter() - start_t) * 1000, 1)

    label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    confidence = round(max(prob, 1 - prob) * 100, 1)

    if prob >= 0.90:
        verdict = "Overwhelmingly Positive"
        emoji = "🤩"
    elif prob >= 0.70:
        verdict = "Strongly Positive"
        emoji = "😊"
    elif prob >= 0.55:
        verdict = "Moderately Positive"
        emoji = "🙂"
    elif prob >= 0.45:
        verdict = "Mixed / Neutral"
        emoji = "😐"
    elif prob >= 0.30:
        verdict = "Moderately Negative"
        emoji = "🙁"
    elif prob >= 0.10:
        verdict = "Strongly Negative"
        emoji = "😠"
    else:
        verdict = "Overwhelmingly Negative"
        emoji = "🤬"

    return jsonify({
        "label": label,
        "probability": round(prob, 4),
        "confidence": confidence,
        "verdict": verdict,
        "emoji": emoji,
        "latency_ms": latency_ms,
    })


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

