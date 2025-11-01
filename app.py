import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib  # <-- FIX 1: Changed from 'pickle' to 'joblib'
import requests
import os
from skimage.feature import hog

# -----------------------------
# 🎯 App Config
# -----------------------------
st.set_page_config(page_title="🫁 Lung Disease Classifier", layout="centered")

st.title("🫁 Lung Disease Classification App")
st.markdown("""
This app uses a **Custom ANN** trained on combined **Pixel + HOG features**
with **SMOTE balancing** and **LIME explainability**.
Upload a **Chest X-ray** to predict whether it shows signs of lung disease.
""")

# -----------------------------
# 📦 Model + Preprocessing Loaders
# -----------------------------

MODEL_URL = "https://huggingface.co/lakshyalol/customann1/resolve/main/lung_disease_model.h5"
MODEL_PATH = "lung_disease_model.h5"
SCALER_PATH = "feature_scaler.pkl"
# ENCODER_PATH = "label_encoder.pkl" # <-- No longer needed, logic is hard-coded

@st.cache_resource
def load_model():
    """Download and load the Keras model."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Hugging Face..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_scaler():
    # Make sure 'feature_scaler.pkl' was created by your training script!
    return joblib.load(SCALER_PATH)

# @st.cache_resource
# def load_label_encoder():
#     return joblib.load(ENCODER_PATH) # <-- This is not needed

# Load once and cache
model = load_model()
scaler = load_scaler()
# label_encoder = load_label_encoder() # <-- Not needed

# -----------------------------
# 📤 File Upload
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))

    # Display image
    st.image(img_resized, caption="🩻 Uploaded X-ray", use_container_width=True, clamp=True)

    # -----------------------------
    # 🧠 Feature Extraction
    # -----------------------------
    img_norm = img_resized.astype("float32") / 255.0
    img_flat = img_norm.flatten()
    img_hog = hog(img_norm, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False)
    combined = np.hstack((img_flat, img_hog)).reshape(1, -1)

    combined_scaled = scaler.transform(combined)

    # -----------------------------
    # 🔮 Prediction
    # -----------------------------
    pred = model.predict(combined_scaled)[0][0]

    # --- FIX 2: INVERTED PREDICTION LOGIC ---
    # The model was trained with 'Lung_Disease' = 0 and 'Normal' = 1
    # So, a high value (like 0.9) means 'Normal'.
    # A low value (like 0.1) means 'Lung Disease'.
    
    if pred > 0.55:
        label = "Normal"
        confidence = pred
    else:
        label = "Lung Disease"
        confidence = 1 - pred

    # -----------------------------
    # 📊 Display Results
    # -----------------------------
    st.markdown("---")
    st.subheader(f"🧪 Prediction: **{label}**")
    st.metric("Confidence", f"{confidence * 100:.2f}%")
else:
    st.info("Please upload a chest X-ray image to begin.")