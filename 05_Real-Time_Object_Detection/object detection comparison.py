import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Create simple image dataset
# -------------------------------
X = np.zeros((300, 32, 32, 3), dtype="float32")
y = np.zeros(300, dtype="int32")

# Create simple shapes
for i in range(300):
    if i < 150:
        # Vertical white line
        X[i, :, 14:18, :] = 1
        y[i] = 0
    else:
        # Horizontal white line
        X[i, 14:18, :, :] = 1
        y[i] = 1

# -------------------------------
# 2. CNN
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(
        16, (3,3), padding="same", activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        32, (5,5), padding="same", activation="relu"
    ),

    tf.keras.layers.AveragePooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 3. Train
# -------------------------------
history = model.fit(
    X, y,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# -------------------------------
# 4. Accuracy
# -------------------------------
loss, accuracy = model.evaluate(X, y, verbose=0)

print("\nAccuracy:", accuracy)

# -------------------------------
# 5. Accuracy graph
# -------------------------------
plt.figure(figsize=(7,5))

plt.plot(history.history["accuracy"], marker="o",
         label="Training Accuracy")

plt.plot(history.history["val_accuracy"], marker="o",
         label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_graph.png", dpi=150,
            bbox_inches="tight")

plt.show()

# -------------------------------
# 6. Feature Map
# -------------------------------
feature_model = tf.keras.Sequential([
    tf.keras.Input(shape=(32,32,3)),
    model.layers[0]
])

feature = feature_model.predict(X[:1], verbose=0)

plt.figure(figsize=(6,5))

plt.imshow(feature[0, :, :, 0], cmap="viridis")
plt.colorbar()
plt.title("CNN Feature Map")
plt.axis("off")

plt.savefig("feature_map.png",
            dpi=150,
            bbox_inches="tight")

plt.show()

# -------------------------------
# 7. Display original image
# -------------------------------
plt.figure(figsize=(5,5))

plt.imshow(X[0])
plt.title("Input Image")
plt.axis("off")

plt.savefig("input_image.png",
            dpi=150,
            bbox_inches="tight")

plt.show()

print("\nPractical completed!")
print("Generated:")
print("1. input_image.png")
print("2. accuracy_graph.png")
print("3. feature_map.png")