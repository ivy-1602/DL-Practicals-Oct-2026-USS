"""
Deep Learning Lab-2 : PRAC 3
Implementation and study of Gradient Problems, Generalization,
and Regularization — Fashion-MNIST Dataset

Steps (per syllabus):
  1. Prepare dataset (normalize, train/val/test split)
  2. Train deep MLP with poor (naive small random) initialization
     -> show vanishing/exploding gradients, plot gradient norms per layer
  3. Apply Xavier and He initialization -> compare gradient plots
  4. Regularization: dropout (p=0.5) + L1/L2 penalties -> monitor val accuracy
  5. Early stopping (patience=5) + data augmentation -> compare val error trends
  6. Batch Normalization vs Layer Normalization -> compare convergence & accuracy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers, callbacks

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# 1. Prepare the dataset
# ---------------------------------------------------------------------

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# flatten for MLP
X_train_full = X_train_full.reshape(len(X_train_full), -1)
X_test = X_test.reshape(len(X_test), -1)

# train / validation split
val_size = 5000
X_val, y_val = X_train_full[:val_size], y_train_full[:val_size]
X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

INPUT_DIM = X_train.shape[1]
NUM_CLASSES = 10
EPOCHS = 20


def build_mlp(hidden_units, kernel_init="random_normal", stddev=0.5,
              dropout=0.0, l1=0.0, l2=0.0, norm=None):
    """Build a deep MLP with configurable init, regularization, and normalization."""
    if kernel_init == "naive":
        init = initializers.RandomNormal(mean=0.0, stddev=stddev)
    elif kernel_init == "xavier":
        init = initializers.GlorotUniform()
    elif kernel_init == "he":
        init = initializers.HeNormal()
    else:
        init = "glorot_uniform"

    reg = None
    if l1 > 0 and l2 > 0:
        reg = regularizers.l1_l2(l1=l1, l2=l2)
    elif l1 > 0:
        reg = regularizers.l1(l1)
    elif l2 > 0:
        reg = regularizers.l2(l2)

    model = keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))
    for units in hidden_units:
        model.add(layers.Dense(units, kernel_initializer=init, kernel_regularizer=reg))
        if norm == "batch":
            model.add(layers.BatchNormalization())
        elif norm == "layer":
            model.add(layers.LayerNormalization())
        model.add(layers.Activation("relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", kernel_initializer=init))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


HIDDEN_UNITS = [128, 64, 32, 16]  # a reasonably deep MLP to make gradient issues visible


def compute_gradient_norms(model, X_batch, y_batch):
    """Compute L2 norm of the gradient w.r.t. each Dense layer's kernel."""
    with tf.GradientTape() as tape:
        preds = model(X_batch, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y_batch, preds)
        loss = tf.reduce_mean(loss)

    dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
    kernels = [l.kernel for l in dense_layers]
    grads = tape.gradient(loss, kernels)
    norms = [float(tf.norm(g)) for g in grads]
    return norms


# ---------------------------------------------------------------------
# 2. Train deep MLP with POOR (naive) initialization
# ---------------------------------------------------------------------

print("\n=== Step 2: Naive initialization (small random weights) ===")
model_naive = build_mlp(HIDDEN_UNITS, kernel_init="naive", stddev=0.5)

batch_x = X_train[:128]
batch_y = y_train[:128]
naive_grad_norms = compute_gradient_norms(model_naive, batch_x, batch_y)
print("Gradient norms per layer (before training):", [f"{g:.5f}" for g in naive_grad_norms])

history_naive = model_naive.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)
naive_grad_norms_after = compute_gradient_norms(model_naive, batch_x, batch_y)
print(f"Naive init -> final val acc: {history_naive.history['val_accuracy'][-1]:.3f}")

# ---------------------------------------------------------------------
# 3. Xavier and He initialization
# ---------------------------------------------------------------------

print("\n=== Step 3: Xavier (Glorot) initialization ===")
model_xavier = build_mlp(HIDDEN_UNITS, kernel_init="xavier")
xavier_grad_norms = compute_gradient_norms(model_xavier, batch_x, batch_y)
history_xavier = model_xavier.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)
print(f"Xavier init -> final val acc: {history_xavier.history['val_accuracy'][-1]:.3f}")

print("\n=== Step 3: He initialization ===")
model_he = build_mlp(HIDDEN_UNITS, kernel_init="he")
he_grad_norms = compute_gradient_norms(model_he, batch_x, batch_y)
history_he = model_he.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)
print(f"He init -> final val acc: {history_he.history['val_accuracy'][-1]:.3f}")

# --- Plot: gradient norms per layer, naive vs xavier vs he ---
fig, ax = plt.subplots(figsize=(8, 5))
layer_idx = list(range(1, len(naive_grad_norms) + 1))
ax.plot(layer_idx, naive_grad_norms, "o-", label="Naive init (std=0.5)")
ax.plot(layer_idx, xavier_grad_norms, "o-", label="Xavier init")
ax.plot(layer_idx, he_grad_norms, "o-", label="He init")
ax.set_xlabel("Layer (1 = first hidden layer)")
ax.set_ylabel("Gradient L2 Norm")
ax.set_title("Gradient Norms Across Layers: Initialization Comparison")
ax.set_yscale("log")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("prac3_gradient_norms.png", dpi=150)
print("\nSaved plot -> prac3_gradient_norms.png")

# ---------------------------------------------------------------------
# 4. Regularization: dropout + L1/L2
# ---------------------------------------------------------------------

print("\n=== Step 4: Regularization (Dropout + L2) ===")
model_reg = build_mlp(HIDDEN_UNITS, kernel_init="he", dropout=0.5, l2=0.001)
history_reg = model_reg.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)
print(f"Dropout+L2 -> final val acc: {history_reg.history['val_accuracy'][-1]:.3f}")

# baseline (He init, no regularization) for comparison
print("\n=== Baseline: He init, no regularization (for comparison) ===")
model_baseline = build_mlp(HIDDEN_UNITS, kernel_init="he")
history_baseline = model_baseline.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history_baseline.history["loss"], label="Baseline train")
axes[0].plot(history_baseline.history["val_loss"], "--", label="Baseline val")
axes[0].plot(history_reg.history["loss"], label="Dropout+L2 train")
axes[0].plot(history_reg.history["val_loss"], "--", label="Dropout+L2 val")
axes[0].set_title("Loss: Baseline vs Regularized")
axes[0].set_xlabel("Epoch")
axes[0].legend(fontsize=8)

axes[1].plot(history_baseline.history["val_accuracy"], label="Baseline val acc")
axes[1].plot(history_reg.history["val_accuracy"], label="Dropout+L2 val acc")
axes[1].set_title("Validation Accuracy: Baseline vs Regularized")
axes[1].set_xlabel("Epoch")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("prac3_regularization.png", dpi=150)
print("Saved plot -> prac3_regularization.png")

# ---------------------------------------------------------------------
# 5. Early stopping + data augmentation
# ---------------------------------------------------------------------

print("\n=== Step 5: Early Stopping (patience=5) ===")
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model_es = build_mlp(HIDDEN_UNITS, kernel_init="he", dropout=0.3, l2=0.0005)
history_es = model_es.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=50, batch_size=128, callbacks=[early_stop], verbose=0,
)
print(f"Early stopping -> stopped at epoch {len(history_es.history['loss'])}, "
      f"final val acc: {history_es.history['val_accuracy'][-1]:.3f}")

print("\n=== Step 5: Data Augmentation (rotation + flip) ===")
# reshape back to images for augmentation
X_train_img = X_train.reshape(-1, 28, 28, 1)
X_val_img = X_val.reshape(-1, 28, 28, 1)

data_aug = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomFlip("horizontal"),
])

def augmented_generator(X, y, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(1000).batch(batch_size)
    ds = ds.map(lambda x, y: (tf.reshape(data_aug(x, training=True), (-1, INPUT_DIM)), y))
    return ds

model_aug = build_mlp(HIDDEN_UNITS, kernel_init="he", dropout=0.3, l2=0.0005)
train_ds_aug = augmented_generator(X_train_img, y_train)
history_aug = model_aug.fit(
    train_ds_aug, validation_data=(X_val, y_val),
    epochs=EPOCHS, verbose=0,
)
print(f"With augmentation -> final val acc: {history_aug.history['val_accuracy'][-1]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_es.history["val_loss"], label="Early stopping (no aug)")
ax.plot(history_aug.history["val_loss"], label="With augmentation")
ax.set_title("Validation Error: Early Stopping vs Data Augmentation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("prac3_earlystop_augmentation.png", dpi=150)
print("Saved plot -> prac3_earlystop_augmentation.png")

# ---------------------------------------------------------------------
# 6. Batch Normalization vs Layer Normalization
# ---------------------------------------------------------------------

print("\n=== Step 6: Batch Normalization ===")
model_bn = build_mlp(HIDDEN_UNITS, kernel_init="he", norm="batch")
history_bn = model_bn.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)
print(f"BatchNorm -> final val acc: {history_bn.history['val_accuracy'][-1]:.3f}")

print("\n=== Step 6: Layer Normalization ===")
model_ln = build_mlp(HIDDEN_UNITS, kernel_init="he", norm="layer")
history_ln = model_ln.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=128, verbose=0,
)
print(f"LayerNorm -> final val acc: {history_ln.history['val_accuracy'][-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history_bn.history["val_loss"], label="BatchNorm")
axes[0].plot(history_ln.history["val_loss"], label="LayerNorm")
axes[0].plot(history_baseline.history["val_loss"], label="No norm (baseline)")
axes[0].set_title("Validation Loss: Normalization Comparison")
axes[0].set_xlabel("Epoch")
axes[0].legend(fontsize=8)

axes[1].plot(history_bn.history["val_accuracy"], label="BatchNorm")
axes[1].plot(history_ln.history["val_accuracy"], label="LayerNorm")
axes[1].plot(history_baseline.history["val_accuracy"], label="No norm (baseline)")
axes[1].set_title("Validation Accuracy: Normalization Comparison")
axes[1].set_xlabel("Epoch")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("prac3_normalization.png", dpi=150)
print("Saved plot -> prac3_normalization.png")

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------

print("\n=== OVERALL SUMMARY (final validation accuracy) ===")
summary = {
    "Naive init": history_naive.history["val_accuracy"][-1],
    "Xavier init": history_xavier.history["val_accuracy"][-1],
    "He init": history_he.history["val_accuracy"][-1],
    "He + Dropout/L2": history_reg.history["val_accuracy"][-1],
    "Early stopping": history_es.history["val_accuracy"][-1],
    "With augmentation": history_aug.history["val_accuracy"][-1],
    "BatchNorm": history_bn.history["val_accuracy"][-1],
    "LayerNorm": history_ln.history["val_accuracy"][-1],
}
for name, acc in summary.items():
    print(f"{name:20s}: {acc:.3f}")