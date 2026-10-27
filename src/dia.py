
import json
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Plant_leave_diseases_dataset_with_augmentation"
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
RESULTS_DIR = BASE_DIR / "diagnostic_results"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# --------------------------------------------------
# 2. LOAD MODEL AND CLASS NAMES
# --------------------------------------------------

def load_resources():
    print("Loading model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output count does not match class_names.json."
        )

    print("Model loaded successfully.")
    print("Number of classes:", len(class_names))

    return model, class_names


# --------------------------------------------------
# 3. LOAD VALIDATION DATASET
# --------------------------------------------------

def load_validation_data(class_names):
    print("\nLoading validation dataset...")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    val_data = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        interpolation="bilinear"
    )

    if val_data.class_names != class_names:
        raise ValueError(
            "Class order mismatch between dataset and class_names.json."
        )

    # Prefetch for more efficient input processing.
    val_data = val_data.prefetch(tf.data.AUTOTUNE)

    print("Validation dataset loaded.")
    print("Validation batches:", len(val_data))

    return val_data


# --------------------------------------------------
# 4. EVALUATE ALL VALIDATION IMAGES
# --------------------------------------------------

def evaluate_model(model, val_data, class_names):
    print("\nEvaluating the full validation dataset...")

    actual_labels = []
    predicted_labels = []
    all_probabilities = []

    for batch_number, (images, labels) in enumerate(
        val_data, start=1
    ):
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        actual_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())

        if batch_number % 50 == 0 or batch_number == len(val_data):
            print(
                f"Processed {batch_number}/{len(val_data)} batches"
            )

    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(actual_labels, predicted_labels)

    print("\n" + "=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)
    print("Validation images:", len(actual_labels))
    print(
        "Correct predictions:",
        int(np.sum(actual_labels == predicted_labels))
    )
    print(
        "Incorrect predictions:",
        int(np.sum(actual_labels != predicted_labels))
    )
    print(f"Overall accuracy: {accuracy * 100:.2f}%")

    # Per-class precision, recall, F1-score and support.
    report_text = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report_text)

    report_dict = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix.
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names)))
    )

    # Calculate accuracy for each actual class.
    class_accuracies = []

    for index, class_name in enumerate(class_names):
        mask = actual_labels == index
        count = int(np.sum(mask))

        class_accuracy = (
            float(np.mean(predicted_labels[mask] == index))
            if count > 0
            else 0.0
        )

        class_accuracies.append(
            (class_name, class_accuracy, count)
        )

    class_accuracies.sort(key=lambda item: item[1])

    print("\nPER-CLASS ACCURACY (LOWEST FIRST)")
    print("-" * 65)

    for name, class_accuracy, count in class_accuracies:
        print(
            f"{name:55s} "
            f"{class_accuracy * 100:6.2f}% "
            f"({count} images)"
        )

    # Find the most frequent errors.
    mistakes = []

    for actual_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if actual_index == predicted_index:
                continue

            count = int(matrix[actual_index, predicted_index])

            if count:
                mistakes.append((
                    count,
                    class_names[actual_index],
                    class_names[predicted_index]
                ))

    mistakes.sort(reverse=True)

    print("\nTOP 20 MISCLASSIFICATIONS")
    print("-" * 65)

    for count, actual_name, predicted_name in mistakes[:20]:
        print(f"{actual_name} -> {predicted_name}: {count} images")

    return (
        actual_labels,
        predicted_labels,
        all_probabilities,
        report_dict,
        matrix,
        accuracy,
        class_accuracies
    )


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

def save_results(
    class_names,
    actual_labels,
    predicted_labels,
    probabilities,
    report,
    matrix,
    accuracy,
    class_accuracies
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save classification report.
    with open(
        RESULTS_DIR / "classification_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Save confusion matrix.
    np.save(RESULTS_DIR / "confusion_matrix.npy", matrix)

    # Save per-class accuracy.
    with open(
        RESULTS_DIR / "class_accuracy.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Class",
            "Accuracy (%)",
            "Validation image count"
        ])

        for name, class_accuracy, count in class_accuracies:
            writer.writerow([
                name,
                round(class_accuracy * 100, 2),
                count
            ])

    # Save individual predictions.
    with open(
        RESULTS_DIR / "predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Actual class",
            "Predicted class",
            "Correct",
            "Predicted confidence (%)"
        ])

        for actual, predicted, probs in zip(
            actual_labels,
            predicted_labels,
            probabilities
        ):
            writer.writerow([
                class_names[int(actual)],
                class_names[int(predicted)],
                bool(actual == predicted),
                round(float(probs[int(predicted)]) * 100, 2)
            ])

    # Plot confusion matrix.
    fig, ax = plt.subplots(figsize=(24, 24))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            name.replace("___", " - ")
            for name in class_names
        ]
    )

    display.plot(
        ax=ax,
        xticks_rotation="vertical",
        include_values=False,
        colorbar=False
    )

    ax.set_title(
        f"Plant Disease CNN Confusion Matrix\n"
        f"Validation Accuracy: {accuracy * 100:.2f}%"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close(fig)

    print("\nResults saved in:", RESULTS_DIR)
    print(" - classification_report.json")
    print(" - class_accuracy.csv")
    print(" - predictions.csv")
    print(" - confusion_matrix.npy")
    print(" - confusion_matrix.png")


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------

def main():
    model, class_names = load_resources()
    val_data = load_validation_data(class_names)

    results = evaluate_model(
        model,
        val_data,
        class_names
    )

    save_results(
        class_names,
        *results[:6],
        results[6]
    )

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
