# Plant Leaf Disease Detection Using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to identify plant leaf diseases from images. It classifies leaf images into 39 categories, including healthy leaves and leaves affected by different diseases.

The model is built using TensorFlow and Keras, and the application provides a simple interface for uploading a leaf image and viewing the predicted class.

## Features
- Image-based plant disease classification
- CNN-based deep learning model
- Supports 39 plant disease and healthy-leaf categories
- Image upload and prediction through a Streamlit web interface
- Displays the top 3 predicted classes with confidence scores

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Pillow (PIL)
- Streamlit

## Project Structure

```text
Plant-leaf-disease-detection/
├── models/
│   └── class_names.json
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── train.py
├── app.py
├── .gitignore
└── requirements.txt
```

*Note: The trained model and dataset are not included in this repository.*

## Dataset
The project uses a plant leaf image dataset containing approximately 61,486 images across 39 classes. The images are organized into folders according to their respective plant disease or healthy-leaf categories.

## How It Works
1. The dataset is loaded and divided into training and validation sets.
2. Images are resized and prepared for training.
3. A CNN learns visual patterns from the leaf images.
4. The trained model predicts the class of an uploaded image.
5. The application displays the predicted categories and their confidence scores.

## Installation

Clone the repository:

```bash
git clone https://github.com/Varun-Agrawal01/Plant-leaf-disease-detection.git
cd Plant-leaf-disease-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python -m streamlit run app.py
```

**Important:** The trained model file must be available at the path expected by `app.py`. Since the model is not included in this repository, download or train the model before running predictions.

## Training the Model

After downloading and extracting the dataset into the expected directory, run:

```bash
python src/train.py
```

Make sure the dataset path and model save path in the source files match your local setup.

## Limitations
- Predictions depend on the quality and characteristics of the input image.
- Images captured in real-world environments may produce less reliable results than dataset images.
- The model's predictions are not a substitute for expert agricultural diagnosis.

## Future Improvements
- Improve performance on real-world leaf images.
- Explore transfer learning with pretrained CNN architectures.
- Add more robust image preprocessing and model evaluation.
- Expand testing across different lighting conditions and backgrounds.

## Author
**Varun Agrawal**

AI Engineering Student
