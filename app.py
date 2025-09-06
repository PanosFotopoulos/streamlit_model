import streamlit as st
import cv2
import numpy as np
import time
import json
import os
import random
from utils import get_image_files, preprocess_image

import tensorflow as tf

# --- Configuration ---
MODEL_PATH = "Sequential2_best_model.h5"
LABEL_DECODER_PATH = "label_decoder.json"
IMAGE_DIR = "test1"  # folder with your test images

# Load labels
try:
    with open(LABEL_DECODER_PATH, "r", encoding="utf-8") as f:
        label_decoder = json.load(f)  # expects {"0":"cat","1":"dog",...}
    CLASS_NAMES = [label_decoder[str(i)] for i in range(len(label_decoder))]
except FileNotFoundError:
    st.error(f"Labels file '{LABEL_DECODER_PATH}' not found. Please create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading or parsing labels from '{LABEL_DECODER_PATH}': {e}")
    st.stop()

st.set_page_config(page_title="Cat or Dog Detector", layout="wide")
st.title("Pet Detector")
st.write("Cat and dog binary classification for Deep Learning.")

# Load Keras model
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading Keras model '{MODEL_PATH}': {e}")
    st.stop()

# Try to show the model input shape (optional)
try:
    MODEL_INPUT_SHAPE = tuple(model.input_shape)
except Exception:
    MODEL_INPUT_SHAPE = None

# Get available images
image_files = get_image_files(IMAGE_DIR)
if not image_files:
    st.error(f"No image files found in '{IMAGE_DIR}'. Please add .jpg/.png images.")
    st.stop()

# Session state for selected image
if 'selected_image_basename' not in st.session_state:
    st.session_state.selected_image_basename = os.path.basename(image_files[0])

selected_image_basename = st.selectbox(
    "Select Image File:",
    options=[os.path.basename(f) for f in image_files],
    key='image_selector',
    index=[os.path.basename(f) for f in image_files].index(st.session_state.selected_image_basename)
      if st.session_state.selected_image_basename in [os.path.basename(f) for f in image_files] else 0
)

if selected_image_basename != st.session_state.selected_image_basename:
    st.session_state.selected_image_basename = selected_image_basename

# Random image button
if st.button("Select Random Image 🎲"):
    if image_files:
        random_image_path = random.choice(image_files)
        st.session_state.selected_image_basename = os.path.basename(random_image_path)
        st.rerun()
    else:
        st.warning("No images available to select randomly.")

selected_image_path = next((f for f in image_files if os.path.basename(f) == st.session_state.selected_image_basename), None)

current_image = None
if selected_image_path:
    current_image = cv2.imread(selected_image_path)
    if current_image is None:
        st.error(f"Error: Could not open image file '{selected_image_path}'.")
        st.stop()
    st.info(f"File '{st.session_state.selected_image_basename}' loaded.")
else:
    st.error("Invalid image file selection.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Image Feed")
    st_image_display = st.empty()

with col2:
    st.subheader("Prediction Probabilities")
    bar_containers = {}
    for class_name in CLASS_NAMES:
        bar_containers[class_name] = st.progress(0, text=f"{class_name}: 0%")

if current_image is not None:
    with col1:
        st_image_display.image(current_image, channels="BGR", use_container_width=True)

    # Preprocess for Keras model: (1, 60, 80, 1) float32 in [0,1]
    x = preprocess_image(current_image)

    # Inference
    try:
        pred = model.predict(x, verbose=0)  # shape could be (1, num_classes) or (1,1) for sigmoid
        probs = pred[0]

        # Handle different output shapes:
        # - If model returns a single sigmoid (binary), expand to two-class probs using label order.
        if probs.ndim == 0:  # scalar
            probs = np.array([1.0 - float(probs), float(probs)], dtype=np.float32)
        elif probs.shape[0] == 1 and len(CLASS_NAMES) == 2:
            p1 = float(probs[0])
            probs = np.array([1.0 - p1, p1], dtype=np.float32)

        # Softmax guard if not normalized
        if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
            exps = np.exp(probs - np.max(probs))
            probs = exps / np.sum(exps)

        # Update UI
        with col2:
            for i, class_name in enumerate(CLASS_NAMES):
                prob = float(probs[i])
                bar_containers[class_name].progress(
                    value=prob,
                    text=f"{class_name}: {int(prob * 100)}%"
                )

        # Top prediction
        top_idx = int(np.argmax(probs))
        st.success(f"Predicted: **{CLASS_NAMES[top_idx]}**  ({probs[top_idx]:.2%})")

    except Exception as e:
        st.warning(f"Error during Keras inference: {e}")

else:
    st.warning("No image selected or loaded for analysis.")
