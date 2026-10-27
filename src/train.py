
import json
import shutil
from pathlib import Path

import tensorflow as tf

from data_loader import load_data
from model import create_model


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

OLD_MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
BACKUP_MODEL_PATH = MODELS_DIR / "plant_disease_model_old.keras"
NEW_MODEL_PATH = MODELS_DIR / "plant_disease_model_v2.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

EPOCHS = 8


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the current model before training.
    if OLD_MODEL_PATH.exists() and not BACKUP_MODEL_PATH.exists():
        shutil.copy2(OLD_MODEL_PATH, BACKUP_MODEL_PATH)
        print("Existing model backed up successfully.")

    elif BACKUP_MODEL_PATH.exists():
        print("Existing backup found; keeping it unchanged.")

    else:
        print("No previous model found to back up.")

    # Load training and validation data.
    print("\nLoading dataset...")
    train_data, val_data, class_names = load_data()

    print("\nNumber of classes:", len(class_names))
    print("Training batches:", len(train_data))
    print("Validation batches:", len(val_data))

    # Save class names in model output order.
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)

    # Create the new CNN.
    print("\nCreating improved CNN...")
    model = create_model(len(class_names))
    model.summary()

    # Save the best checkpoint and stop when validation stops improving.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(NEW_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=0.00001,
            mode="min",
            verbose=1
        )
    ]

    # Train the new model.
    print("\nStarting training...")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nTraining completed.")

    # Report the best validation accuracy seen during training.
    best_epoch = (
        history.history["val_loss"].index(
            min(history.history["val_loss"])
        ) + 1
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    print("\nTraining summary:")
    print("Best validation-loss epoch:", best_epoch)
    print(
        f"Highest validation accuracy during training: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print("\nNew model saved at:")
    print(NEW_MODEL_PATH)

    print("\nOriginal model backup:")
    print(BACKUP_MODEL_PATH)

    print("\nThe existing app model has not been replaced.")


if __name__ == "__main__":
    main()
