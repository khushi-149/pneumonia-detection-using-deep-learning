import os
import tensorflow as tf

from data_preprocessing import create_data_generators
from model import build_model


# Training settings
EPOCHS = 10
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")


def train_model():

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load training, validation, and test data
    train_data, val_data, test_data = create_data_generators()

    # Build the VGG16 model
    model = build_model()

    # Display model architecture
    model.summary()

    # Save the best model during training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Stop training if validation performance stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    print("Training completed!")

    return model, history


if __name__ == "__main__":
    train_model()
