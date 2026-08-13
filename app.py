import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# --------------------------------------------------
# Title and Introduction
# --------------------------------------------------

st.title("🫁 Chest X-ray Pneumonia Detection")
st.subheader("CNN-Based Medical Image Classification")

st.write(
    """
    Upload a chest X-ray image and the trained Convolutional Neural Network
    will classify it as **NORMAL** or **PNEUMONIA**.
    """
)

st.info(
    "This application is an educational and research demonstration. "
    "It is not intended to provide medical diagnosis or replace evaluation "
    "by a qualified healthcare professional."
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "best_pneumonia_cnn.keras"


@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_pneumonia_model()
except Exception as e:
    st.error(
        "The trained CNN model could not be loaded. "
        "Please make sure 'best_pneumonia_cnn.keras' is in the same "
        "folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

IMG_SIZE = (224, 224)


def predict_pneumonia(image):
    """
    Preprocess an uploaded chest X-ray and return
    the predicted class and pneumonia probability.
    """

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    if probability >= 0.5:
        predicted_class = "PNEUMONIA"
    else:
        predicted_class = "NORMAL"

    return predicted_class, probability


# --------------------------------------------------
# File Upload
# --------------------------------------------------

st.markdown("### 📤 Upload a Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("### 🖼️ Uploaded X-ray")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width=500
    )

    if st.button("🔍 Analyze X-ray", use_container_width=True):

        with st.spinner("Analyzing X-ray..."):

            predicted_class, probability = predict_pneumonia(image)

        st.markdown("### 🧠 Prediction Result")

        if predicted_class == "PNEUMONIA":

            st.error("🫁 Prediction: PNEUMONIA")

            st.metric(
                "Pneumonia Probability",
                f"{probability:.2%}"
            )

        else:

            st.success("✅ Prediction: NORMAL")

            st.metric(
                "Pneumonia Probability",
                f"{probability:.2%}"
            )

        st.progress(probability)

        st.caption(
            "The probability shown is the model's output score and "
            "should not be interpreted as a clinical diagnosis."
        )

# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.markdown("---")

st.markdown("### 📊 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Model:** Convolutional Neural Network")
    st.write("**Input Size:** 224 × 224 pixels")
    st.write("**Classes:** Normal / Pneumonia")

with col2:
    st.write("**Test ROC-AUC:** 95.66%")
    st.write("**Pneumonia Recall:** 98.72%")
    st.write("**Test Accuracy:** 81.57%")

st.markdown("---")


st.caption(
    "Chest X-ray Pneumonia Detection | CNN Deep Learning Project"
)

