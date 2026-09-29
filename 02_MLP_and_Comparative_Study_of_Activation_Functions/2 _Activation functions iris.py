"""
Deep Learning Lab-2 : PRAC 2
Implementation of Multilayer Perceptron and Comparative Study of
Activation Functions — IRIS Dataset

Steps (per syllabus):
  1. Load and normalize the dataset
  2. Build an MLP (one hidden layer) using Keras
  3. Train separate models using: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU
  4. Compare: training loss, validation accuracy, convergence speed
  5. Plot activation function curves themselves
  6. Interpret the effect of non-linearity on learning
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load and normalize the dataset
# ---------------------------------------------------------------------

iris = load_iris()
X, y = iris.data, iris.target  # 150 samples, 4 features, 3 classes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {iris.target_names}")

# ---------------------------------------------------------------------
# 2 & 3. Build + train an MLP (one hidden layer) per activation function
# ---------------------------------------------------------------------

activations = {
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "relu": "relu",
    "leaky_relu": layers.LeakyReLU(negative_slope=0.1),
    "elu": "elu",
}

results = {}
histories = {}

for name, act in activations.items():
    print(f"\n=== Training MLP with {name} activation ===")

    model = keras.Sequential(name=f"mlp_{name}")
    model.add(layers.Input(shape=(4,)))
    if isinstance(act, str):
        model.add(layers.Dense(16, activation=act))
    else:
        # LeakyReLU is a layer, not a string activation
        model.add(layers.Dense(16))
        model.add(act)
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=8,
        verbose=0,
    )
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # convergence speed: first epoch where val_accuracy reaches within 2% of its final value
    val_acc = history.history["val_accuracy"]
    final_acc = val_acc[-1]
    convergence_epoch = next(
        (i for i, a in enumerate(val_acc) if a >= final_acc - 0.02), len(val_acc)
    )

    results[name] = {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "time": elapsed,
        "convergence_epoch": convergence_epoch,
    }
    histories[name] = history.history

    print(f"{name:12s} -> test acc: {test_acc:.3f} | test loss: {test_loss:.3f} "
          f"| converged ~epoch {convergence_epoch} | time: {elapsed:.2f}s")

# ---------------------------------------------------------------------
# 4. Comparison plots: loss curves, val accuracy curves
# ---------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for name, hist in histories.items():
    axes[0, 0].plot(hist["loss"], label=name)
    axes[0, 1].plot(hist["val_loss"], label=name)
    axes[1, 0].plot(hist["val_accuracy"], label=name)

axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend(fontsize=8)

axes[0, 1].set_title("Validation Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend(fontsize=8)

axes[1, 0].set_title("Validation Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].legend(fontsize=8)

# final test accuracy bar chart
names = list(results.keys())
accs = [results[n]["test_acc"] for n in names]
axes[1, 1].bar(names, accs, color=["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#8d99ae"])
axes[1, 1].set_title("Final Test Accuracy by Activation")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("prac2_training_curves.png", dpi=150)
print("\nSaved plot -> prac2_training_curves.png")

# ---------------------------------------------------------------------
# 5. Plot the activation function curves themselves
# ---------------------------------------------------------------------

x = np.linspace(-5, 5, 200)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x, alpha=0.1): return np.where(x > 0, x, alpha * x)
def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(x) - 1))

plt.figure(figsize=(8, 6))
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")
plt.plot(x, elu(x), label="ELU")
plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.title("Activation Function Shapes")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("prac2_activation_shapes.png", dpi=150)
print("Saved plot -> prac2_activation_shapes.png")

# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------
print("\n=== SUMMARY ===")
print(f"{'Activation':12s} {'Test Acc':>10s} {'Test Loss':>10s} {'Conv. Epoch':>12s} {'Time(s)':>10s}")
for name in names:
    r = results[name]
    print(f"{name:12s} {r['test_acc']:>10.3f} {r['test_loss']:>10.3f} {r['convergence_epoch']:>12d} {r['time']:>10.2f}")