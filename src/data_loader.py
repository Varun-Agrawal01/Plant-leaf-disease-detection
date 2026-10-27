
import tensorflow as tf

DATA_DIR = (
    r"D:\final\Plant_leave_diseases_dataset_with_augmentation"
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def load_data():
    train_data = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_data.class_names

    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)

    print("Number of classes:", len(class_names))
    print("Classes:", class_names)

    return train_data, val_data, class_names
if __name__ == "__main__":
    train_data, val_data, class_names = load_data()

    print("Number of classes:", len(class_names))

    images, labels = next(iter(train_data))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)