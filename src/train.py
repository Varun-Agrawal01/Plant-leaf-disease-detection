
import json
from pathlib import Path

from data_loader import load_data
from model import create_model


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

EPOCHS = 10


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    train_data, val_data, class_names = load_data()

    print("Number of classes:", len(class_names))
    print("Training batches:", len(train_data))
    print("Validation batches:", len(val_data))

    # Save class names
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)

    # Create model
    print("\nCreating CNN model...")
    model = create_model(len(class_names))
    model.summary()

    # Train model
    print("\nStarting training...")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    # Save trained model
    model.save(MODEL_PATH)

    print("\nTraining completed!")
    print("Model saved at:", MODEL_PATH)

    best_val_accuracy = max(history.history["val_accuracy"])
    print(
        f"Highest validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
