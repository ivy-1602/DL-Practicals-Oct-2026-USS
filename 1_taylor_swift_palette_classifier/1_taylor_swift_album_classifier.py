"""
Deep Learning Lab-2 : PRAC 1 (adapted)
Implementation of Backpropagation in Multilayer Neural Networks
for Image Classification — Taylor Swift Album Palette Dataset

Structure mirrors the syllabus prac (originally MNIST digits):
  1. Load and preprocess image dataset
  2. Build deep MLPs with 1, 2, and 3 hidden layers
  3. Train using backpropagation (Keras autodiff)
  4. Softmax output layer
  5. Compare against Logistic Regression and Decision Tree
  6. Analyze accuracy, loss curves, training time
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "data/augmented"
IMG_SIZE = (64, 64)
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load and preprocess dataset
# ---------------------------------------------------------------------

def load_dataset(data_dir):
    classes = sorted(os.listdir(data_dir))
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            img = Image.open(os.path.join(cls_dir, fname)).convert("RGB").resize(IMG_SIZE)
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(cls)
    return np.array(X), np.array(y), classes


print("Loading dataset...")
X, y_labels, class_names = load_dataset(DATA_DIR)
print(f"Loaded {len(X)} images, {len(class_names)} classes: {class_names}")

le = LabelEncoder()
y = le.fit_transform(y_labels)
num_classes = len(class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Flattened versions for MLP / sklearn baselines
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

# ---------------------------------------------------------------------
# 2 & 3. Build + train deep MLPs (1, 2, 3 hidden layers) via backprop
# ---------------------------------------------------------------------

def build_mlp(num_hidden_layers, input_dim, num_classes, units=128):
    model = keras.Sequential(name=f"mlp_{num_hidden_layers}hidden")
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(0.3))
        units = max(units // 2, 16)
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


input_dim = X_train_flat.shape[1]
results = {}
histories = {}

for n_hidden in [1, 2, 3]:
    print(f"\n=== Training MLP with {n_hidden} hidden layer(s) ===")
    model = build_mlp(n_hidden, input_dim, num_classes)

    start = time.time()
    history = model.fit(
        X_train_flat, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=16,
        verbose=0,
    )
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
    results[f"MLP-{n_hidden}H"] = {"accuracy": test_acc, "time": elapsed}
    histories[f"MLP-{n_hidden}H"] = history.history

    print(f"MLP-{n_hidden}H  ->  test acc: {test_acc:.3f}  |  train time: {elapsed:.2f}s")

# ---------------------------------------------------------------------
# 4. Baseline comparisons: Logistic Regression, Decision Tree
# ---------------------------------------------------------------------

print("\n=== Baseline: Logistic Regression ===")
start = time.time()
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_flat, y_train)
lr_time = time.time() - start
lr_acc = accuracy_score(y_test, logreg.predict(X_test_flat))
results["LogisticRegression"] = {"accuracy": lr_acc, "time": lr_time}
print(f"LogReg -> test acc: {lr_acc:.3f} | train time: {lr_time:.2f}s")

print("\n=== Baseline: Decision Tree ===")
start = time.time()
dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
dtree.fit(X_train_flat, y_train)
dt_time = time.time() - start
dt_acc = accuracy_score(y_test, dtree.predict(X_test_flat))
results["DecisionTree"] = {"accuracy": dt_acc, "time": dt_time}
print(f"DecisionTree -> test acc: {dt_acc:.3f} | train time: {dt_time:.2f}s")

# ---------------------------------------------------------------------
# 5. Analysis: accuracy, loss curves, training time
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Accuracy comparison bar chart
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
axes[0].bar(names, accs, color=["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#8d99ae"])
axes[0].set_title("Test Accuracy by Model")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=30)

# (b) Loss curves for the 3 MLPs
for name, hist in histories.items():
    axes[1].plot(hist["loss"], label=f"{name} train")
    axes[1].plot(hist["val_loss"], "--", label=f"{name} val")
axes[1].set_title("MLP Loss Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=8)

# (c) Training time comparison
times = [results[n]["time"] for n in names]
axes[2].bar(names, times, color=["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#8d99ae"])
axes[2].set_title("Training Time (s)")
axes[2].set_ylabel("Seconds")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("results_summary.png", dpi=150)
print("\nSaved plot -> results_summary.png")

# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------
print("\n=== SUMMARY ===")
print(f"{'Model':20s} {'Accuracy':>10s} {'Time (s)':>10s}")
for name in names:
    print(f"{name:20s} {results[name]['accuracy']:>10.3f} {results[name]['time']:>10.2f}")
