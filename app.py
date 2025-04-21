import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
from skimage.feature import hog

# load model and preprocessing tools
model = tf.keras.models.load_model("lung_disease_model.h5")

with open("feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.set_page_config(page_title="Lung Disease Classifier", layout="centered")

st.title("🫁 Lung Disease Classification App")
st.markdown(
    "Built using a **Custom ANN** trained on combined **Pixel + HOG features** with **SMOTE** and **LIME Explainability** (optional backend)."
)

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    
    st.image(img_resized, caption="Uploaded X-ray", use_column_width=True, clamp=True)

    # Normalize
    img_norm = img_resized.astype("float32") / 255.0

    # Extract features
    img_flat = img_norm.flatten()
    img_hog = hog(img_norm, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    combined = np.hstack((img_flat, img_hog)).reshape(1, -1)

    # Scale
    combined_scaled = scaler.transform(combined)

    # Predict
    pred = model.predict(combined_scaled)[0][0]
    label = "Lung Disease" if pred > 0.55 else "Normal"
    confidence = pred if pred > 0.55 else 1 - pred

    st.markdown(f"### 🧪 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence: **{confidence*100:.2f}%**")