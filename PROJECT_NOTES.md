# IMDB Sentiment Classification (RNN → Bi-LSTM) — Key Terms, Examples & Viva Questions

## 📌 Project Overview

A movie-review sentiment classifier built with TensorFlow/Keras, trained on a balanced 20k sample of the IMDB 50k dataset. The notebook shows the full journey: a **SimpleRNN** baseline → a **regularized SimpleRNN** → a final **Bidirectional LSTM** that reaches **~85.8% test accuracy** (test loss 0.361). A Flask web app lets you try the model live.

Pipeline: raw review → TextVectorization → Embedding → SpatialDropout1D → Bidirectional LSTM → Dense(sigmoid) → probability.

---

## 🔑 Key Terms (with examples)

### 1. Simple RNN (Recurrent Neural Network)
Processes text one word at a time while carrying a hidden state (memory) of earlier words.
- **Example:** for *"not good"*, it reads "not", then "good" with memory of "not" → NEGATIVE.

### 2. LSTM (Long Short-Term Memory)
A smarter recurrent cell with gates (forget/input/output) that can remember or discard information over long stretches of text.
- **Example:** in *"I really liked the story even though the ending was slow... loved it"*, an LSTM can keep the early "liked" in memory across the sentenceand still finish POSITIVE.

### 3. Bidirectional LSTM
Reads the sequence **both forwards and backwards**, so each word's representation sees both its left and right context.
- **Example:** in *"... was boring, the sequel was great"*, the word "great" gets context from what comes *after* it too — better sentiment cues.

### 4. Embedding
A lookup table converting each word ID into a dense vector (here 64 numbers) that captures meaning.
- **Example:** *"good" → [0.2, -0.5, 0.9, ...]*, and similar words end up close together.

### 5. TextVectorization
Keras layer that lowercases text, splits into words, and maps each to an ID.
- **Example:** `"I Love this"` → `[37, 91, 15]`; reviews are truncated/padded to 200 tokens; only the top 10,000 words are kept.

### 6. Vocabulary & `[UNK]`
The learned word list from the **training split only**; out-of-vocabulary words map to one special `[UNK]` token.
- **Example:** *"fantasticizzle"* → `[UNK]` — the model still runs instead of crashing.

### 7. Padding & `mask_zero`
Sequences are padded with 0s to length 200; `mask_zero=True` tells the Embedding layer to ignore them.
- **Example:** `[17, 42, 8, 0, 0, ...]` — the trailing 0s are skipped.

### 8. Dropout
Randomly switches off a fraction of units during training so the model can't over-rely on any one feature — a core regularization technique.
- **Example:** LSTM with `dropout=0.3` trains faster to generalize and slows overfitting.

### 9. Recurrent Dropout
Dropout applied to the connections *between* time steps of an RNN/LSTM — it regularizes the memory path.
- **Example:** `recurrent_dropout=0.3` prevents the model from memorizing a specific word-to-word pattern.

### 10. SpatialDropout1D
Drops entire feature maps (all time steps of one dimension) instead of individual elements — popular for text because it removes whole word channels.
- **Example:** with rate 0.3, roughly 30% of embedding channels are zeroed each forward pass.

### 11. L2 Regularization
Adds a penalty proportional to the squared size of the weights, keeping them small and discouraging over-complex functions.
- **Example:** `kernel_regularizer=keras.regularizers.l2(0.001)`.

### 12. Learning Rate
The step size the optimizer uses to update weights. Too big → unstable loss; too small → slow/imprecise training.
- **Example:** baseline Adam used default lr; the improved RNN used `learning_rate=0.0005` to stabilize training; the final Bi-LSTM used lr `0.001`.

### 13. Epochs, Batch Size & Early Stopping
- **Epoch:** one full pass over the training data; the final model trains up to **15 epochs**.
- **Batch size 128:** weights are updated after every 128 reviews.
- **EarlyStopping(patience=4/5, restore_best_weights=True):** stop when val_loss stops improving and reload the best weights — the built-in cure for overfitting.

### 14. Loss vs Accuracy
- **Loss (binary cross-entropy):** how wrong the probabilities are — lower is better (final: **0.361**).
- **Accuracy:** fraction correctly classified (final: **~85.8%**).

### 15. Sigmoid & the 0.5 Threshold
Sigmoid maps the raw score to (0, 1); probability ≥ 0.5 → POSITIVE, else NEGATIVE.
- **Example:** `0.984` → POSITIVE (great review), `0.001` → NEGATIVE (terrible review).

### 16. Overfitting
Memorizing the training set instead of learning general patterns (train acc ~98% but val ~77%). The notebook fights it with dropout, recurrent dropout, SpatialDropout1D, L2, lower learning rate, and early stopping.

### 17. Stratified Train/Val/Test Split (70/10/20)
Splits that keep the same positive/negative balance as the original data so evaluation isn't skewed.
- **Example:** `stratify=y` on 4,000-review test set keeps ~50/50.

---

## ❓ Questions a Teacher May Ask (with answers)

**Q1. What problem does this project solve?**
A: Binary sentiment classification of raw IMDB movie reviews into POSITIVE or NEGATIVE.

**Q2. What was your final model and how accurate is it?**
A: A Bidirectional LSTM (64 units, dropout 0.3, SpatialDropout1D 0.3, L2 regularization). Test loss 0.361, test accuracy ~85.8%.

**Q3. Why evolve from SimpleRNN to Bi-LSTM?**
A: SimpleRNN suffered from overfitting and instability. Regularization helped, but ultimately an LSTM's gated memory handles long-range context better, and the bidirectional pass sees text both ways — so both accuracy and stability improved.

**Q4. How does raw text become numbers?**
A: `TextVectorization` lowercases, tokenizes, and IDs each word; the Embedding layer maps IDs to 64-D vectors; padding to length 200 with `mask_zero` on the Embedding.

**Q5. Why is the vocabulary limited to 10,000?**
A: The most frequent 10k words cover nearly all reviews; the rare tail only adds noise and overfitting risk.

**Q6. What does dropout do here?**
A: It randomly disables ~30% of units (and 30% of recurrent connections) during training, forcing the model to not rely on any single feature/pattern — the main defense against overfitting.

**Q7. What is SpatialDropout1D and why use it for text?**
A: It drops whole embedding channels (entire word-dimensions) rather than random elements — text-specific regularization that removes redundant word features.

**Q8. Why use L2 regularization on the Dense layer?**
A: It penalizes large weights, encouraging smoother decision rules and reducing overfitting.

**Q9. Why binary cross-entropy?**
A: It's the standard loss for a two-class probability from a sigmoid output; it sharply penalizes confident wrong answers.

**Q10. Why did you lower the Adam learning rate at first?**
A: A smaller lr (5e-4) made the SimpleRNN's training more stable; for the final Bi-LSTM, lr 0.001 with early stopping worked better.

**Q11. What is overfitting and how did you detect/fix it?**
A: Train accuracy hit ~98% while validation plateaued ~77–82% — classic overfitting. Fixed with dropout, SpatialDropout1D, L2, and EarlyStopping with `restore_best_weights=True`.

**Q12. Why a Bidirectional LSTM over a normal LSTM?**
A: Each token's representation considers both left and right context, so the sentiment signal (often carried by later words like "but... loved it") is captured better.

**Q13. How did you split the data and why stratify?**
A: 70/10/20 train/val/test with `stratify=y` so each split keeps roughly the same positive/negative ratio — otherwise metrics would be skewed.

**Q14. Why does the web app use `weights.h5` + `vocab.json` instead of a `.keras` file?**
A: Keras 3 can't reliably serialize a `TextVectorization` layer whose vocabulary contains an empty-string token. Saving weights and the vocabulary separately and rebuilding the architecture gives identical predictions.

**Q15. How does the web app work?**
A: Flask loads the weights + vocabulary, rebuilds the Bi-LSTM via `model_def.build_model`, and serves `/api/predict`. The frontend posts the review text and animates the returned label, probability, and confidence.

**Q16. What could further improve it?**
A: Pretrained embeddings (GloVe), GRU/CNN hybrids, or fine-tuning a transformer such as DistilBERT; bigger balanced data and better text cleaning also help.

**Q17. Is the model biased or limited?**
A: It saw only 20k of 50k reviews, truncates to 200 words, and sends rare words to `[UNK]`, so very unusual phrasing can be misclassified.
