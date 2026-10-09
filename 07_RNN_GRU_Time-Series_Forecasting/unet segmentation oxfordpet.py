"""
Deep Learning Lab-2 : PRAC 6
Semantic Segmentation using U-Net — Oxford-IIIT Pet Dataset

Steps:
load dataset -> build U-Net -> pixel-wise classification ->
visualize predicted vs ground-truth masks ->
evaluate IoU, Dice, pixel accuracy.

Install once:
pip install tensorflow tensorflow-datasets matplotlib numpy
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMG_SIZE = 128
BATCH = 32
EPOCHS = 15


# ---------------------------------------------------------------------------
# 1. Load and preprocess dataset
# ---------------------------------------------------------------------------

def preprocess(sample):
    # Resize image and normalize pixel values to [0, 1]
    img = tf.image.resize(
        sample["image"],
        (IMG_SIZE, IMG_SIZE)
    ) / 255.0

    # Resize segmentation mask using nearest-neighbor
    # so that class labels are not interpolated
    mask = tf.image.resize(
        sample["segmentation_mask"],
        (IMG_SIZE, IMG_SIZE),
        method="nearest"
    )

    # Oxford-IIIT Pet labels are 1, 2, 3
    # Convert them to 0, 1, 2
    mask = tf.cast(mask, tf.int32) - 1

    return img, mask


# Oxford-IIIT Pet currently available as version 4.0.0
ds_train, info = tfds.load(
    "oxford_iiit_pet:4.0.0",
    split="train",
    with_info=True
)

ds_test = tfds.load(
    "oxford_iiit_pet:4.0.0",
    split="test"
)

# Preprocess, shuffle, batch and prefetch
train = (
    ds_train
    .map(preprocess)
    .shuffle(500)
    .batch(BATCH)
    .prefetch(1)
)

test = (
    ds_test
    .map(preprocess)
    .batch(BATCH)
    .prefetch(1)
)


# ---------------------------------------------------------------------------
# 2. U-Net model
# ---------------------------------------------------------------------------

def conv_block(x, filters):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x


def build_unet(input_size=IMG_SIZE, num_classes=3):

    inputs = layers.Input(
        (input_size, input_size, 3)
    )

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    bottleneck = conv_block(p3, 256)

    # Decoder
    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(bottleneck)

    u3 = conv_block(
        layers.Concatenate()([u3, c3]),
        128
    )

    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(u3)

    u2 = conv_block(
        layers.Concatenate()([u2, c2]),
        64
    )

    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(u2)

    u1 = conv_block(
        layers.Concatenate()([u1, c1]),
        32
    )

    # Output layer
    outputs = layers.Conv2D(
        num_classes,
        1,
        activation="softmax"
    )(u1)

    return Model(
        inputs,
        outputs,
        name="U-Net"
    )


# Create model
model = build_unet()

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()


# ---------------------------------------------------------------------------
# 3. Train the model
# ---------------------------------------------------------------------------

history = model.fit(
    train,
    validation_data=test,
    epochs=EPOCHS
)


# ---------------------------------------------------------------------------
# 4. Evaluate IoU, Dice and Pixel Accuracy
# ---------------------------------------------------------------------------

def compute_metrics(model, dataset, num_classes=3):

    ious = []
    dices = []
    pixel_accs = []

    for imgs, masks in dataset:

        # Predict segmentation classes
        predictions = model.predict(
            imgs,
            verbose=0
        )

        preds = tf.argmax(
            predictions,
            axis=-1,
            output_type=tf.int32
        )

        # Remove mask channel dimension
        masks = tf.squeeze(
            masks,
            axis=-1
        )

        # Calculate IoU and Dice for each class
        for c in range(num_classes):

            pred_c = tf.cast(
                preds == c,
                tf.float32
            )

            true_c = tf.cast(
                masks == c,
                tf.float32
            )

            intersection = tf.reduce_sum(
                pred_c * true_c
            )

            union = (
                tf.reduce_sum(pred_c)
                + tf.reduce_sum(true_c)
                - intersection
            )

            iou = intersection / (
                union + 1e-7
            )

            dice = (
                2 * intersection
            ) / (
                tf.reduce_sum(pred_c)
                + tf.reduce_sum(true_c)
                + 1e-7
            )

            ious.append(float(iou))
            dices.append(float(dice))

        # Pixel accuracy
        pixel_accuracy = tf.reduce_mean(
            tf.cast(
                preds == masks,
                tf.float32
            )
        )

        pixel_accs.append(
            float(pixel_accuracy)
        )

    return (
        np.mean(ious),
        np.mean(dices),
        np.mean(pixel_accs)
    )


# Calculate metrics
mean_iou, mean_dice, pixel_acc = compute_metrics(
    model,
    test
)

print(
    f"\nMean IoU: {mean_iou:.3f}"
    f" | Mean Dice: {mean_dice:.3f}"
    f" | Pixel Accuracy: {pixel_acc:.3f}"
)


# ---------------------------------------------------------------------------
# 5. Visualize predictions vs ground truth
# ---------------------------------------------------------------------------

sample_imgs, sample_masks = next(
    iter(test)
)

# Predict first 4 images
preds = tf.argmax(
    model.predict(
        sample_imgs[:4],
        verbose=0
    ),
    axis=-1
)


# Create visualization
fig, axes = plt.subplots(
    3,
    4,
    figsize=(12, 9)
)


for i in range(4):

    # Original image
    axes[0, i].imshow(
        sample_imgs[i]
    )

    axes[0, i].set_title(
        "Image"
    )

    axes[0, i].axis("off")


    # Ground truth mask
    axes[1, i].imshow(
        sample_masks[i, :, :, 0],
        cmap="viridis"
    )

    axes[1, i].set_title(
        "Ground Truth"
    )

    axes[1, i].axis("off")


    # Predicted mask
    axes[2, i].imshow(
        preds[i],
        cmap="viridis"
    )

    axes[2, i].set_title(
        "Predicted"
    )

    axes[2, i].axis("off")


plt.tight_layout()

plt.savefig(
    "prac6_segmentation_results.png",
    dpi=150
)

plt.close()

print(
    "Saved plot -> prac6_segmentation_results.png"
)


# ---------------------------------------------------------------------------
# 6. Plot training loss and accuracy
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4)
)


# Loss curve
axes[0].plot(
    history.history["loss"],
    label="train"
)

axes[0].plot(
    history.history["val_loss"],
    label="val"
)

axes[0].set_title(
    "Loss"
)

axes[0].legend()


# Accuracy curve
axes[1].plot(
    history.history["accuracy"],
    label="train"
)

axes[1].plot(
    history.history["val_accuracy"],
    label="val"
)

axes[1].set_title(
    "Pixel Accuracy"
)

axes[1].legend()


plt.tight_layout()

plt.savefig(
    "prac6_training_curves.png",
    dpi=150
)

plt.close()

print(
    "Saved plot -> prac6_training_curves.png"
)