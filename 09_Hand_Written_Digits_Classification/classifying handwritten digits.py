# ============================================================
# PRACTICAL 10
# Handwritten Digit Classification using MNIST
# + REST API using Flask
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)

# ============================================================
# 2. NORMALIZE PIXEL VALUES
# ============================================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ============================================================
# 3. CREATE SEQUENTIAL NEURAL NETWORK
# ============================================================

model = tf.keras.Sequential([
    
    tf.keras.layers.Flatten(
        input_shape=(28, 28)
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully!")

model.summary()

# ============================================================
# 4. TRAIN MODEL FOR 5 EPOCHS
# ============================================================

print("\nTraining model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ============================================================
# 5. EVALUATE MODEL
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print("Test Loss     :", round(test_loss, 4))
print("Test Accuracy :", round(test_accuracy * 100, 2), "%")

# ============================================================
# 6. SAVE TRAINED MODEL
# ============================================================

model.save("mnist_digit_model.keras")

print("\nModel saved as:")
print("mnist_digit_model.keras")

# ============================================================
# 7. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

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

plt.title("MNIST Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "mnist_accuracy.png",
    dpi=200
)

plt.show()

# ============================================================
# 8. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("MNIST Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "mnist_loss.png",
    dpi=200
)

plt.show()

# ============================================================
# 9. SAMPLE PREDICTIONS
# ============================================================

predictions = model.predict(
    x_test[:10],
    verbose=0
)

predicted_digits = np.argmax(
    predictions,
    axis=1
)

print("\nSample Predictions:")
print("Actual   :", y_test[:10])
print("Predicted:", predicted_digits)

# ============================================================
# 10. DISPLAY SAMPLE IMAGES
# ============================================================

plt.figure(figsize=(10, 4))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[i],
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[i]}\nPred: {predicted_digits[i]}"
    )

    plt.axis("off")

plt.suptitle("MNIST Digit Predictions")
plt.tight_layout()

plt.savefig(
    "mnist_predictions.png",
    dpi=200
)

plt.show()

# ============================================================
# 11. CREATE FLASK REST API
# ============================================================

app = Flask(__name__)

# Load saved model
api_model = tf.keras.models.load_model(
    "mnist_digit_model.keras"
)

# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "MNIST Digit Classification REST API",
        "status": "running",
        "endpoint": "/predict"
    })


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "error": "Please upload an image using key 'image'"
            }), 400

        image_file = request.files["image"]

        # Read image bytes
        image_bytes = image_file.read()

        # Decode image using TensorFlow
        image = tf.io.decode_image(
            image_bytes,
            channels=1,
            expand_animations=False
        )

        # Resize to MNIST size
        image = tf.image.resize(
            image,
            [28, 28]
        )

        # Convert to float
        image = tf.cast(
            image,
            tf.float32
        )

        # Normalize
        image = image / 255.0

        # Add batch dimension
        image = tf.expand_dims(
            image,
            axis=0
        )

        # Predict
        prediction = api_model.predict(
            image,
            verbose=0
        )

        digit = int(
            np.argmax(prediction)
        )

        confidence = float(
            np.max(prediction)
        )

        return jsonify({
            "predicted_digit": digit,
            "confidence": round(
                confidence * 100,
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 12. START REST API
# ============================================================

print("\n" + "=" * 50)
print("REST API STARTING")
print("=" * 50)

print("Open in browser:")
print("http://127.0.0.1:5000")

print("\nPrediction endpoint:")
print("POST http://127.0.0.1:5000/predict")

print("\nPress CTRL+C to stop the server.")

app.run(
    host="127.0.0.1",
    port=5000,
    debug=False
)