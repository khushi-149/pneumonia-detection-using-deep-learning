# Pneumonia Detection from Chest X-Rays Using Deep Learning

A deep learning project that classifies chest X-ray images into **NORMAL** and **PNEUMONIA** categories using Transfer Learning with VGG16.

## Project Overview

Pneumonia is a lung infection that can be identified through medical imaging. This project uses a Convolutional Neural Network (CNN) with a pre-trained VGG16 model to classify chest X-ray images.

It also includes a Streamlit web application where users can upload an X-ray image and view the model's prediction.

## Features

- Chest X-ray image classification
- Transfer learning using VGG16 (ImageNet weights)
- Image preprocessing and prediction
- Model evaluation using classification metrics
- Interactive Streamlit web application

## Technologies Used

- Python
- TensorFlow and Keras
- VGG16
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains two classes:

- NORMAL
- PNEUMONIA

The training data was divided into training and validation sets.

## Model Architecture

The project uses VGG16 for transfer learning.

- Pre-trained VGG16 convolutional base
- Frozen convolutional layers
- Global Average Pooling
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (0.5)
- Sigmoid output layer for binary classification

**Training configuration:**
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: Binary Crossentropy
- Epochs: Up to 8
- Input image size: 224 × 224

## Model Performance

The model achieved **86% test accuracy** on 624 test images.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| NORMAL | 0.89 | 0.71 | 0.79 |
| PNEUMONIA | 0.84 | 0.95 | 0.89 |

**Overall accuracy: 86%**

## Project Structure

```text
pneumonia-detection/
│
├── app.py
├── requirements.txt
├── models/
│   └── pneumonia_model.keras
│
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── outputs/
├── data/
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/khushi-149/pneumonia-detection-using-deep-learning.git
```

### 2. Navigate to the project directory

```bash
cd pneumonia-detection-using-deep-learning
```

### 3. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Upload a chest X-ray image and click **Predict** to view the result.

## Disclaimer

This project is developed for educational and learning purposes only. It is not intended to provide medical diagnoses or replace professional medical advice.

## Author

Developed as a deep learning and computer vision project using Python, TensorFlow/Keras, and VGG16.
