import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array


MODEL_PATH = "models/pneumonia_model.keras"
IMG_SIZE = (224, 224)


def predict_image(image_path):

    # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please train the model first."
        )

    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load and resize the X-ray image
    image = load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    # Convert image into a numerical array
    image_array = img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Apply VGG16 preprocessing
    image_array = tf.keras.applications.vgg16.preprocess_input(
        image_array
    )

    # Predict
    probability = model.predict(image_array, verbose=0)[0][0]

    # Display result
    if probability >= 0.5:
        print("Prediction: PNEUMONIA")
    else:
        print("Prediction: NORMAL")

    print(f"Pneumonia score: {probability:.4f}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path_to_xray_image")
    else:
        predict_image(sys.argv[1])
