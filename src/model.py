import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models


def build_model(input_shape=(224, 224, 3)):
    """
    Build a pneumonia classification model using VGG16.
    Classes: NORMAL and PNEUMONIA.
    """

    # Load pre-trained VGG16 without its original classifier
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze VGG16 layers initially
    base_model.trainable = False

    # Build our own classification layers
    model = models.Sequential([
        layers.Input(shape=input_shape),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    # Configure the model for binary classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
