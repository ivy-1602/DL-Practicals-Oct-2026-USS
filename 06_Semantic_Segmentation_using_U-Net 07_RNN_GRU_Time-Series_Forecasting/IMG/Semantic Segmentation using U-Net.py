import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PRACTICAL 7: SEMANTIC SEGMENTATION USING U-NET
# Zero download / Zero new installation version
# ============================================================

IMG = 64
N = 40

# ------------------------------------------------------------
# 1. Create simple pet-like images and segmentation masks
# ------------------------------------------------------------

X = np.zeros((N, IMG, IMG, 3), dtype="float32")
Y = np.zeros((N, IMG, IMG, 1), dtype="float32")

for i in range(N):

    # Background
    X[i] = 0.15

    # Pet body
    cx = np.random.randint(27, 38)
    cy = np.random.randint(27, 38)

    yy, xx = np.ogrid[:IMG, :IMG]

    body = ((xx-cx)**2 / 18**2 +
            (yy-cy)**2 / 23**2) < 1

    # Ears
    ear1 = ((xx-(cx-11))**2 +
            (yy-(cy-20))**2) < 8**2

    ear2 = ((xx-(cx+11))**2 +
            (yy-(cy-20))**2) < 8**2

    pet = body | ear1 | ear2

    # RGB pet
    X[i][pet] = [0.75, 0.45, 0.25]

    # Mask
    Y[i][pet, 0] = 1


# ------------------------------------------------------------
# 2. U-NET
# ------------------------------------------------------------

inputs = tf.keras.Input(
    shape=(IMG, IMG, 3)
)

# Encoder
c1 = tf.keras.layers.Conv2D(
    8, 3, activation="relu", padding="same"
)(inputs)

c1 = tf.keras.layers.Conv2D(
    8, 3, activation="relu", padding="same"
)(c1)

p1 = tf.keras.layers.MaxPooling2D()(c1)


c2 = tf.keras.layers.Conv2D(
    16, 3, activation="relu", padding="same"
)(p1)

c2 = tf.keras.layers.Conv2D(
    16, 3, activation="relu", padding="same"
)(c2)

p2 = tf.keras.layers.MaxPooling2D()(c2)


# Bottleneck
b = tf.keras.layers.Conv2D(
    32, 3, activation="relu", padding="same"
)(p2)


# Decoder
u1 = tf.keras.layers.UpSampling2D()(b)

u1 = tf.keras.layers.Concatenate()([
    u1, c2
])

c3 = tf.keras.layers.Conv2D(
    16, 3, activation="relu", padding="same"
)(u1)


u2 = tf.keras.layers.UpSampling2D()(c3)

u2 = tf.keras.layers.Concatenate()([
    u2, c1
])

c4 = tf.keras.layers.Conv2D(
    8, 3, activation="relu", padding="same"
)(u2)


# Output
outputs = tf.keras.layers.Conv2D(
    1, 1, activation="sigmoid"
)(c4)


model = tf.keras.Model(
    inputs,
    outputs
)


# ------------------------------------------------------------
# 3. Compile
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 4. Train
# ------------------------------------------------------------

history = model.fit(
    X,
    Y,
    epochs=5,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)


# ------------------------------------------------------------
# 5. Prediction
# ------------------------------------------------------------

prediction = model.predict(
    X[:1],
    verbose=0
)

pred_mask = (
    prediction[0] > 0.5
).astype("float32")


# ------------------------------------------------------------
# 6. Metrics
# ------------------------------------------------------------

true = Y[0].flatten()
pred = pred_mask.flatten()

intersection = np.sum(true * pred)

union = (
    np.sum(true)
    + np.sum(pred)
    - intersection
)

iou = intersection / (union + 1e-7)

dice = (
    2 * intersection
) / (
    np.sum(true)
    + np.sum(pred)
    + 1e-7
)

pixel_accuracy = np.mean(
    true == pred
)


print("\n==============================")
print("SEGMENTATION RESULTS")
print("==============================")

print("IoU              :", round(iou, 4))
print("Dice Coefficient :", round(dice, 4))
print("Pixel Accuracy   :", round(pixel_accuracy, 4))


# ------------------------------------------------------------
# 7. Original / Ground Truth / Prediction
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)

plt.imshow(X[0])

plt.title("Input Image")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    Y[0].squeeze(),
    cmap="gray"
)

plt.title("Ground Truth Mask")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    pred_mask.squeeze(),
    cmap="gray"
)

plt.title("Predicted Mask")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "segmentation_result.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 8. Accuracy Graph
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "U-Net Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "accuracy_graph.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 9. Metrics Graph
# ------------------------------------------------------------

metrics = [
    "IoU",
    "Dice",
    "Pixel Accuracy"
]

values = [
    iou,
    dice,
    pixel_accuracy
]

plt.figure(figsize=(7, 5))

plt.bar(
    metrics,
    values
)

plt.ylim(0, 1)

plt.ylabel("Score")

plt.title(
    "U-Net Segmentation Performance"
)

plt.grid(
    axis="y"
)

plt.savefig(
    "metrics_graph.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print("\n==============================")
print("PRACTICAL COMPLETED")
print("==============================")

print("Generated:")
print("• segmentation_result.png")
print("• accuracy_graph.png")
print("• metrics_graph.png")