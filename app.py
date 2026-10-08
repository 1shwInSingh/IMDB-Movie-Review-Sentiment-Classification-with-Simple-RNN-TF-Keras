"""Flask web frontend for the IMDB Bidirectional LSTM sentiment classifier."""
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from model_def import build_model

app = Flask(__name__)

WEIGHTS_PATH = "imdb_sentiment_bilstm.weights.h5"
VOCAB_PATH = "vocab.json"
METRICS_PATH = "metrics.json"

with open(VOCAB_PATH, encoding="utf-8") as f:
    vocab_list = json.load(f)

model = build_model(vocab_list)
model.load_weights(WEIGHTS_PATH)

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics, metrics_json=json.dumps(metrics))


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    import time
    start_t = time.perf_counter()
    data = request.get_json(force=True)
    review = (data.get("review") or "").strip()
    if not review:
        return jsonify({"error": "Please enter a review."}), 400
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
