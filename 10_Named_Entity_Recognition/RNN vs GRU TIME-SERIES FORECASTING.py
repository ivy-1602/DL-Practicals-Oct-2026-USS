import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# ============================================================
# PRACTICAL 8 : RNN vs GRU TIME-SERIES FORECASTING
# ============================================================

np.random.seed(10)
tf.random.set_seed(10)

# ------------------------------------------------------------
# 1. Create stock-like historical data
# ------------------------------------------------------------

days = 500

t = np.arange(days)

# Simulated stock price with trend + seasonality + noise
price = (
    100
    + 0.05 * t
    + 5 * np.sin(t / 15)
    + 2 * np.sin(t / 40)
    + np.random.normal(0, 1, days)
)

# ------------------------------------------------------------
# 2. Visualize stock trend
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(price)

plt.title("Simulated Stock Price Trend")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.grid(True)

plt.savefig(
    "stock_trend.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 3. Normalize using Min-Max Scaling
# ------------------------------------------------------------

minimum = price.min()
maximum = price.max()

scaled = (price - minimum) / (maximum - minimum)

# ------------------------------------------------------------
# 4. Create 30-day sliding windows
# ------------------------------------------------------------

window = 30

X = []
y = []

for i in range(len(scaled) - window):
    X.append(scaled[i:i + window])
    y.append(scaled[i + window])

X = np.array(X)
y = np.array(y)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# ------------------------------------------------------------
# 5. Train/Test split
# ------------------------------------------------------------

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ------------------------------------------------------------
# 6. RNN Model
# ------------------------------------------------------------

rnn = tf.keras.Sequential([
    tf.keras.Input(shape=(30, 1)),
    tf.keras.layers.SimpleRNN(50),
    tf.keras.layers.Dense(1)
])

rnn.compile(
    optimizer="adam",
    loss="mse"
)

start = time.time()

rnn_history = rnn.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

rnn_time = time.time() - start

# ------------------------------------------------------------
# 7. GRU Model
# ------------------------------------------------------------

gru = tf.keras.Sequential([
    tf.keras.Input(shape=(30, 1)),
    tf.keras.layers.GRU(50),
    tf.keras.layers.Dense(1)
])

gru.compile(
    optimizer="adam",
    loss="mse"
)

start = time.time()

gru_history = gru.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

gru_time = time.time() - start

# ------------------------------------------------------------
# 8. Predictions
# ------------------------------------------------------------

rnn_pred = rnn.predict(
    X_test,
    verbose=0
).flatten()

gru_pred = gru.predict(
    X_test,
    verbose=0
).flatten()

# Convert back to original price
actual = y_test * (maximum - minimum) + minimum

rnn_pred = rnn_pred * (maximum - minimum) + minimum

gru_pred = gru_pred * (maximum - minimum) + minimum

# ------------------------------------------------------------
# 9. Prediction Graph
# ------------------------------------------------------------

plt.figure(figsize=(11, 5))

plt.plot(
    actual,
    label="Actual Price"
)

plt.plot(
    rnn_pred,
    label="RNN Prediction"
)

plt.plot(
    gru_pred,
    label="GRU Prediction"
)

plt.title(
    "RNN vs GRU Stock Price Prediction"
)

plt.xlabel("Test Days")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True)

plt.savefig(
    "prediction_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 10. Training Loss Graph
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    rnn_history.history["loss"],
    marker="o",
    label="RNN Training Loss"
)

plt.plot(
    rnn_history.history["val_loss"],
    marker="o",
    label="RNN Validation Loss"
)

plt.plot(
    gru_history.history["loss"],
    marker="s",
    label="GRU Training Loss"
)

plt.plot(
    gru_history.history["val_loss"],
    marker="s",
    label="GRU Validation Loss"
)

plt.title(
    "RNN vs GRU Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "loss_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 11. Calculate RMSE
# ------------------------------------------------------------

rnn_rmse = np.sqrt(
    np.mean((actual - rnn_pred) ** 2)
)

gru_rmse = np.sqrt(
    np.mean((actual - gru_pred) ** 2)
)

# ------------------------------------------------------------
# 12. Performance Comparison
# ------------------------------------------------------------

models = [
    "RNN",
    "GRU"
]

rmse = [
    rnn_rmse,
    gru_rmse
]

training_time = [
    rnn_time,
    gru_time
]

plt.figure(figsize=(7, 5))

plt.bar(
    models,
    rmse
)

plt.ylabel("RMSE")
plt.title("RNN vs GRU Prediction Error")
plt.grid(axis="y")

plt.savefig(
    "rmse_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 13. Results
# ------------------------------------------------------------

print("\n================================")
print("RNN vs GRU RESULTS")
print("================================")

print("RNN RMSE :", round(rnn_rmse, 3))
print("GRU RMSE :", round(gru_rmse, 3))

print("\nRNN Training Time :",
      round(rnn_time, 2), "seconds")

print("GRU Training Time :",
      round(gru_time, 2), "seconds")

print("\nGenerated files:")
print("1. stock_trend.png")
print("2. prediction_comparison.png")
print("3. loss_comparison.png")
print("4. rmse_comparison.png")