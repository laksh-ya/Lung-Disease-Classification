import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
import requests
import os
from skimage.feature import hog, local_binary_pattern

# ===========================
# STREAMLIT CONFIG
# ===========================
st.set_page_config(page_title="🫁 Lung Disease Classifier", layout="wide")

st.title("🫁 Lung Disease Classification App")
st.markdown("""
This app uses a **Custom ANN** trained on **Pixel + HOG features**  
with **SMOTE balancing** and **LIME explainability**.

Below, we also show what the model 'sees' using:
- **Sharpening**
- **Histogram Equalization**
- **LBP**
- **HOG Visualization**
- **SIFT Keypoints**
""")

# ===========================
# MODEL + SCALER LOADING
# ===========================
MODEL_URL = "https://huggingface.co/lakshyalol/customann1/resolve/main/lung_disease_model.h5"
MODEL_PATH = "lung_disease_model.h5"
SCALER_PATH = "feature_scaler.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model file..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# ===========================
# FILE UPLOAD
# ===========================
uploaded_file = st.file_uploader("📸 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload a chest X-ray to begin.")
    st.stop()

# Read image into grayscale
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (128, 128))
img_norm = img_resized.astype("float32") / 255.0

st.image(img_resized, caption="🩻 Uploaded X-ray", use_container_width=True, clamp=True)


# ============================================================
# 1️⃣ PREPROCESSING: SHOWCASE SECTION
# ============================================================

st.markdown("---")
st.subheader("🔍 Preprocessing & Feature Showcase")

col1, col2, col3 = st.columns(3)

# -------------------------
# SHARPENING
# -------------------------
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(img_resized, -1, kernel)

with col1:
    st.caption("🗡️ Sharpened Image")
    st.image(sharpened, use_container_width=True, clamp=True)

# -------------------------
# HISTOGRAM EQUALIZATION
# -------------------------
heq = cv2.equalizeHist(img_resized)

with col2:
    st.caption("📈 Histogram Equalization")
    st.image(heq, use_container_width=True, clamp=True)

# -------------------------
# LBP (Local Binary Patterns)
# -------------------------
lbp = local_binary_pattern(img_resized, P=8, R=1, method='uniform')

with col3:
    st.caption("🌐 LBP Texture Map")
    st.image(lbp, use_container_width=True, clamp=True)


# ============================================================
# 2️⃣ ADVANCED FEATURE VISUALIZATION
# ============================================================

col4, col5 = st.columns(2)

# -------------------------
# HOG + Visualization
# -------------------------
hog_features, hog_image = hog(
    img_norm,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

with col4:
    st.caption("📐 HOG Visualization")
    st.image(hog_image, use_container_width=True, clamp=True)

# -------------------------
# SIFT Keypoints
# -------------------------
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img_resized, None)
kp_img = cv2.drawKeypoints(
    img_resized,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

with col5:
    st.caption("🔑 SIFT Keypoints")
    st.image(kp_img, use_container_width=True, clamp=True)


# ============================================================
# 3️⃣ MODEL FEATURE EXTRACTION (UNCHANGED)
# ============================================================

img_flat = img_norm.flatten()
img_hog = hog_features
combined = np.hstack((img_flat, img_hog)).reshape(1, -1)
combined_scaled = scaler.transform(combined)

# ============================================================
# 4️⃣ PREDICTION
# ============================================================

pred = model.predict(combined_scaled)[0][0]

if pred > 0.55:
    label = "Normal"
    confidence = pred
else:
    label = "Lung Disease"
    confidence = 1 - pred

st.markdown("---")
st.subheader(f"🧪 Prediction: **{label}**")
st.metric("Confidence", f"{confidence * 100:.2f}%")