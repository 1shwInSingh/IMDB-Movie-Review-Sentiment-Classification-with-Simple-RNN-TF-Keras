# IMDB Review Sentiment Classification (TF/Keras) + Web Viewer

A movie-review sentiment classifier built in TensorFlow/Keras that classifies a raw IMDB review as **POSITIVE** or **NEGATIVE**. The project goes from a **SimpleRNN** baseline to a regularized **Bidirectional LSTM**, and ships with a Flask web app where you can paste a review and see the model's prediction live.

## The Journey in the Notebook

| Stage | Model | Notes |
|-------|-------|-------|
| Baseline | SimpleRNN (32 units) | Quick to train, but overfits and is unstable |
| Improved | SimpleRNN + SpatialDropout1D, dropout 0.2, L2, Adam lr=5e-4 | Slower to overfit, better generalization |
| Final | **Bidirectional LSTM (64 units)** + dropout 0.3, SpatialDropout1D 0.3, L2 | **Test accuracy ≈ 85.8%** |

## Project Structure

```
├── IMDB Review Classification With LSTM.ipynb   # the full project notebook
├── app.py                                       # Flask web server
├── model_def.py                                 # shared model definition (Bi-LSTM)
├── train_model.py                               # trains the final Bi-LSTM and saves artifacts
├── metrics.json                                 # test metrics (after training)
├── vocab.json                                   # vocabulary used by the web app
├── templates/index.html                         # frontend
├── static/style.css                             # frontend styling
└── PROJECT_NOTES.md                             # key terms + viva Q&A
```

## Getting Started

```bash
pip install -r requirements.txt
```

### 1. Train the model

```bash
PYTHONUTF8=1 python train_model.py
```

This downloads nothing — it reads `IMDB Dataset.csv` locally, trains the Bidirectional LSTM (up to 15 epochs with early stopping), and writes `imdb_sentiment_bilstm.weights.h5`, `vocab.json`, and `metrics.json`.

### 2. Run the web app

```bash
PYTHONUTF8=1 python app.py
```

Open **http://127.0.0.1:5000/** and type a review.

> **Note:** `PYTHONUTF8=1` is needed on Windows so the vocabulary (which may contain non-ASCII characters like en-dashes) serializes correctly.

## Results

- Test loss: **0.361** &middot; Test accuracy: **≈ 85.8%**
- Positive sample: `POSITIVE (0.984)` &middot; Negative sample: `NEGATIVE (0.001)`

## Requirements

See `requirements.txt` — TensorFlow 2.x, scikit-learn, pandas, numpy, Flask.

## License

MIT — see `LICENSE`.
