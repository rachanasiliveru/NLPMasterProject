# -*- coding: utf-8 -*-

"""
Homework 3: Perceptron-based spam classifier on the SMS Spam Collection.

- Reuses evaluation functions from homework1.py:
  confusion_matrix, accuracy, precision, recall, f1_score
- Uses the same SMS dataset approach as Homework 2, but keeps the code
  self-contained here for clarity.
- Builds a binary bag-of-words vocabulary from the training set only.
- Trains a perceptron (spam = +1, normal = -1).
- Evaluates on the test set using Homework 1 metrics.
"""

import random
import urllib.request
import os
import zipfile
import io

# Reuse evaluation metrics from homework1.py
from homework1 import confusion_matrix, accuracy, precision, recall, f1_score


# =========================================================
# STEP 1: LOAD REAL SMS DATASET
# =========================================================

def load_sms_dataset(filepath="SMSSpamCollection"):
    if not os.path.exists(filepath):
        print("Dataset not found locally. Downloading...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        with urllib.request.urlopen(url) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            with z.open("SMSSpamCollection") as f:
                content = f.read().decode("utf-8")

        with open(filepath, "w", encoding="utf-8") as out:
            out.write(content)

        print("Download complete. Saved as 'SMSSpamCollection'.")
    else:
        print("Dataset found locally.")

    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            label_raw, text = parts[0], parts[1]

            # Perceptron labels: spam = +1, normal = -1
            label = 1 if label_raw == "spam" else -1
            dataset.append((text, label))

    return dataset


def train_test_split(dataset, test_ratio=0.2, seed=42):
    data = dataset[:]
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


# =========================================================
# STEP 2: PERCEPTRON UTILITIES
# =========================================================

def tokenize(text):
    return text.lower().split()


def preview_tokenization(train_set, limit=20):
    """
    Show tokenization preview only for the first few training samples.
    Tokenization is still done for the full training set in later steps.
    """
    print("\n=== Step 3: Tokenization Preview (train set) ===")
    for text, label in train_set[:limit]:
        label_name = "spam" if label == 1 else "normal"
        print(f"{label_name:6} | '{text[:40]}' -> {tokenize(text)}")

    if len(train_set) > limit:
        print(f"\nTokenization is done for the full training set,")
        print(f"but only the first {limit} samples are shown because the dataset is large.")


def build_vocab(dataset):
    vocab = {}
    idx = 0
    for text, _ in dataset:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def vectorize(text, vocab):
    vector = [0] * len(vocab)
    for token in tokenize(text):
        if token in vocab:
            vector[vocab[token]] = 1
    return vector


def predict(x):
    score = sum(w[i] * x[i] for i in range(len(x))) + b
    # Count score <= 0 as normal
    return 1 if score > 0 else -1


def update(x, y):
    global w, b
    if predict(x) != y:
        w = [w[i] + y * x[i] for i in range(len(w))]
        b = b + y
        return 1   # one mistake happened
    return 0       # no mistake


def to_binary_label(y):
    """
    Convert labels from {-1, 1} to {0, 1}
    for homework1 evaluation functions.
    """
    return 1 if y == 1 else 0


def show_misclassified_examples(test_set, vocab, limit=5):
    """
    Show a few misclassified test examples so errors are easy to inspect.
    """
    print("\n=== Extra: Some Misclassified Test Examples ===")
    shown = 0

    for text, label in test_set:
        x = vectorize(text, vocab)
        pred = predict(x)

        if pred != label:
            true_name = "spam" if label == 1 else "normal"
            pred_name = "spam" if pred == 1 else "normal"
            print(f"TEXT: {text[:70]}")
            print(f"TRUE: {true_name} | PRED: {pred_name}")
            print("-" * 60)
            shown += 1

            if shown >= limit:
                break

    if shown == 0:
        print("No misclassifications found in the displayed test examples.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=== Homework 3: Perceptron Spam Classifier (SMS) ===")

    # =========================================================
    # STEP 3: LOAD DATA
    # =========================================================

    print("\n=== Step 1: Load Dataset ===")
    dataset = load_sms_dataset()
    print(f"Total samples: {len(dataset)}")

    print("\n=== Step 2: Train/Test Split ===")
    train_set, test_set = train_test_split(dataset, test_ratio=0.2, seed=42)
    print(f"Training samples : {len(train_set)}")
    print(f"Test samples     : {len(test_set)}")

    preview_tokenization(train_set, limit=20)

    # =========================================================
    # STEP 4: BUILD VOCAB ONLY FROM TRAIN SET
    # =========================================================

    print("\n=== Step 4: Build Vocabulary ===")
    vocab = build_vocab(train_set)
    print(f"Vocabulary size: {len(vocab)}")

    # =========================================================
    # STEP 5: INIT MODEL
    # =========================================================

    w = [0] * len(vocab)
    b = 0

    print("\n=== Step 5: Initialize Model ===")
    print("weights length:", len(w))
    print("bias:", b)

    # =========================================================
    # STEP 6: TRAIN UNTIL NO MISTAKES OR MAX EPOCHS
    # =========================================================

    print("\n=== Step 6: Training ===")

    max_epochs = 20
    for epoch in range(max_epochs):
        mistakes = 0

        for text, label in train_set:
            x = vectorize(text, vocab)
            mistakes += update(x, label)

        print(f"Epoch {epoch + 1}: mistakes = {mistakes}")

        if mistakes == 0:
            print("Training converged. No mistakes in this epoch.")
            break
    else:
        print("Reached maximum epochs without full convergence.")

    # =========================================================
    # STEP 7: TEST ON NEW EXAMPLES
    # =========================================================

    print("\n=== Step 7: Test Example Sentences ===")
    tests = [
        "free money now",
        "call me later",
        "win a prize now",
        "let's meet tomorrow",
        "urgent claim your cash",
        "happy birthday friend"
    ]

    for t in tests:
        x = vectorize(t, vocab)
        pred = predict(x)
        label_name = "spam" if pred == 1 else "normal"
        print(f"'{t}' -> {pred} ({label_name})")

    # =========================================================
    # STEP 8: TEST SET PREDICTIONS
    # =========================================================

    print("\n=== Step 8: Some Predictions on Test Set ===")
    for text, label in test_set[:10]:
        x = vectorize(text, vocab)
        pred = predict(x)

        true_name = "spam" if label == 1 else "normal"
        pred_name = "spam" if pred == 1 else "normal"

        print(f"TEXT: {text[:70]}")
        print(f"TRUE: {true_name} | PRED: {pred_name}")
        print("-" * 60)

    # Show some mistakes so missed cases are visible
    show_misclassified_examples(test_set, vocab, limit=5)

    # =========================================================
    # STEP 9: INSPECT WEIGHTS
    # =========================================================

    print("\n=== Step 9: Inspect Weights ===")
    for word, i in list(vocab.items())[:30]:
        print(f"{word:15s} w={w[i]:+d}")

    print(f"\n{'bias':15s} b={b:+d}")

    # =========================================================
    # STEP 10: EVALUATION WITH HOMEWORK 1 METRICS
    # =========================================================

    print("\n=== Step 10: Evaluation on Test Set (Homework 1 metrics) ===")
    print("Using confusion_matrix, accuracy, precision, recall, f1_score from homework1.py")

    y_true = []
    y_pred = []

    for text, label in test_set:
        x = vectorize(text, vocab)
        pred = predict(x)

        y_true.append(to_binary_label(label))
        y_pred.append(to_binary_label(pred))

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall = {rec:.4f}")
    print(f"F1 Score = {f1:.4f}")

    # Final summary block
    print("\n=== Final Summary ===")
    print("Model: Perceptron with binary bag-of-words features")
    print("Dataset: SMS Spam Collection")
    print(f"Train/Test split: {len(train_set)} / {len(test_set)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Final Accuracy: {acc:.4f}")
    print(f"Final F1 Score: {f1:.4f}")

    print("\n=== End of Homework 3 run ===")