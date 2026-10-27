
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# -------------------- PATHS --------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


# -------------------- APP INTERFACE --------------------

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant leaf image to predict its disease "
    "using a Convolutional Neural Network (CNN)."
)

st.info(
    "For best results, upload a clear image of a single "
    "plant leaf with visible details."
)


# -------------------- CHECK FILES --------------------

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

if not CLASS_NAMES_PATH.exists():
    st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
    st.stop()


# -------------------- LOAD MODEL AND CLASSES --------------------

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Could not load the model or class names: {error}")
    st.stop()


if model.output_shape[-1] != len(class_names):
    st.error(
        "The number of model outputs does not match "
        "the number of class names. Check your model and class_names.json."
    )
    st.stop()


# -------------------- IMAGE UPLOAD --------------------

uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


# -------------------- PREDICTION --------------------

if uploaded_file is not None:

    try:
        # Open image and convert to RGB
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Image selected for prediction",
            width="stretch"
        )

        # Resize using bilinear interpolation
        image = image.resize(
            (128, 128),
            Image.Resampling.BILINEAR
        )

        # Convert image to NumPy array
        image_array = np.asarray(image, dtype=np.float32)

        # Add batch dimension:
        # (128, 128, 3) -> (1, 128, 128, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict probabilities
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Find the highest-probability class
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        # Display result
        st.subheader("Prediction Result")

        st.success(
            f"Predicted class: {predicted_class.replace('___', ' - ').replace('_', ' ')}"
        )

        st.metric(
            label="Model confidence",
            value=f"{confidence:.2f}%"
        )

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee that the diagnosis is correct."
        )

        # Show top 3 predictions
        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(predictions)[-3:][::-1]

        for index in top_indices:
            class_name = class_names[int(index)]
            probability = float(predictions[index]) * 100

            display_name = class_name.replace(
                "___", " - "
            ).replace("_", " ")

            st.write(f"**{display_name}** — {probability:.2f}%")
            st.progress(float(predictions[index]))

        st.warning(
            "This is an educational image-classification project. "
            "Predictions may be incorrect, especially for blurry images, "
            "unfamiliar backgrounds, or diseases not represented in the dataset."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.write("Upload an image above to get started.")

st.divider()
st.caption("Plant Disease Detection | TensorFlow CNN")
