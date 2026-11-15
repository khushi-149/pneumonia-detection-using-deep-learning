import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# Model settings
MODEL_PATH = "models/pneumonia_model.keras"
IMG_SIZE = (224, 224)

# Page title
st.title("🫁 Pneumonia Detection")
st.write("Upload a chest X-ray image to get a model prediction.")

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Chest X-ray", width=350)

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            # Load model
            model = load_model()

            # Prepare image
            image = load_img(
                uploaded_file,
                target_size=IMG_SIZE,
                color_mode="rgb"
            )

            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # VGG16 preprocessing
            image_array = tf.keras.applications.vgg16.preprocess_input(
                image_array
            )

            # Prediction
            probability = float(
                model.predict(image_array, verbose=0)[0][0]
            )

            # Display result
            st.subheader("Prediction Result")

            if probability >= 0.5:
                st.error("Prediction: PNEUMONIA")
            else:
                st.success("Prediction: NORMAL")

            st.write(
                f"Pneumonia score: {probability:.2%}"
            )

            st.caption(
                "Educational project only. Not intended for medical diagnosis."
            )
