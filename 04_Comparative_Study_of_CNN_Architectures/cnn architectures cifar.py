import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create a tiny image dataset automatically
X = np.random.rand(500, 28, 28, 1)
y = np.random.randint(0, 2, 500)

# CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, activation='relu',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=2, verbose=1)

# Prediction
p = model.predict(X[:1])
print("Predicted class:", np.argmax(p))

# Save PNG automatically
plt.imshow(X[0].squeeze(), cmap='gray')
plt.title("CNN Input Image")
plt.axis("off")
plt.savefig("cnn_result.png")
plt.show()