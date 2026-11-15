import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data_preprocessing import create_data_generators


MODEL_PATH = "models/pneumonia_model.keras"


def evaluate_model():

    # Check whether the trained model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load test images
    _, _, test_data = create_data_generators()

    # Predict probabilities
    probabilities = model.predict(test_data)

    # Convert probabilities into class predictions
    predictions = (probabilities >= 0.5).astype(int).ravel()

    # Get actual labels
    actual = test_data.classes

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predictions,
            target_names=["NORMAL", "PNEUMONIA"]
        )
    )

    # Create confusion matrix
    cm = confusion_matrix(actual, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"]
    )

    display.plot(cmap="Blues")
    plt.title("Pneumonia Detection - Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
