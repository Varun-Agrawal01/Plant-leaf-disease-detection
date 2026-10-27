
import tensorflow as tf


def create_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),

        # Normalize pixel values from 0-255 to 0-1
        tf.keras.layers.Rescaling(1.0 / 255),

        # Convolution block 1
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Convolution block 2
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Convolution block 3
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classification layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
