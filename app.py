import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import time
import json
import os
import random # Import the random module
from utils import get_image_files, preprocess_image


# --- Configuration ---
ONNX_MODEL_PATH = "model_quantized.onnx"
LABEL_DECODER_PATH = "label_decoder.json"
IMAGE_DIR = "test" # Changed from VIDEO_DIR to IMAGE_DIR

# Load labels
try:
    with open(LABEL_DECODER_PATH, "r") as f:
        label_decoder = json.load(f)
    CLASS_NAMES = [label_decoder[str(i)] for i in range(len(label_decoder))]
except FileNotFoundError:
    st.error(f"Labels file '{LABEL_DECODER_PATH}' not found. Please create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading or parsing labels from '{LABEL_DECODER_PATH}': {e}")
    st.stop()

# Load ONNX model
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    st.error(f"Error loading ONNX model '{ONNX_MODEL_PATH}': {e}")
    st.stop()

MODEL_INPUT_SHAPE = tuple(session.get_inputs()[0].shape)


st.set_page_config(page_title="Driver Distraction Detector", layout="wide")
st.title("Driver Distraction Detector App")
st.write("Driver distraction classification using Deep Learning.")


image_files = get_image_files(IMAGE_DIR) # Use the new function to get image files
if not image_files:
    st.error(f"No image files found in '{IMAGE_DIR}'. Please add your image files (e.g., .jpg, .png) to this directory.")
    st.stop()

# Initialize session state for selected image if not already set
if 'selected_image_basename' not in st.session_state:
    st.session_state.selected_image_basename = os.path.basename(image_files[0]) if image_files else None

# Selectbox for image selection
selected_image_basename = st.selectbox(
    "Select Image File:",
    options=[os.path.basename(f) for f in image_files],
    key='image_selector', # Use a key to ensure proper state management
    index=[os.path.basename(f) for f in image_files].index(st.session_state.selected_image_basename) if st.session_state.selected_image_basename in [os.path.basename(f) for f in image_files] else 0
)

# Update session state if selectbox value changes
if selected_image_basename != st.session_state.selected_image_basename:
    st.session_state.selected_image_basename = selected_image_basename


# Button to select a random image
if st.button("Select Random Image 🎲"):
    if image_files:
        random_image_path = random.choice(image_files)
        st.session_state.selected_image_basename = os.path.basename(random_image_path)
        st.rerun() # Rerun the app to reflect the new selection
    else:
        st.warning("No images available to select randomly.")


selected_image_path = next((f for f in image_files if os.path.basename(f) == st.session_state.selected_image_basename), None)

current_image = None
if selected_image_path:
    current_image = cv2.imread(selected_image_path)
    if current_image is None:
        st.error(f"Error: Could not open image file '{selected_image_path}'. Make sure it's a valid image format.")
        st.stop()
    st.info(f"File '{st.session_state.selected_image_basename}' loaded.")
else:
    st.error("Invalid image file selection. This should not happen.")
    st.stop()


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Image Feed")
    st_image_display = st.empty() # Changed from st_frame to st_image_display

with col2:
    st.subheader("Prediction Probabilities")
    bar_containers = {}
    for i, class_name in enumerate(CLASS_NAMES):
        bar_containers[class_name] = st.progress(0, text=f"{class_name}: 0%")


if current_image is not None:
    with col1:
        st_image_display.image(current_image, channels="BGR", use_container_width=True)

    # --- Preprocess the ORIGINAL image for ONNX model inference ---
    processed_image = preprocess_image(current_image)

    # Run inference
    try:
        outputs = session.run([output_name], {input_name: processed_image})
        probabilities = outputs[0][0]
        with col2:
            for i, class_name in enumerate(CLASS_NAMES):
                prob = probabilities[i]
                bar_containers[class_name].progress(
                    value=float(prob),
                    text=f"{class_name}: {int(prob*100)}%",
                    width='stretch')

    except Exception as e:
        st.warning(f"Error during ONNX inference: {e}")

else:
    st.warning("No image selected or loaded for analysis.")


st.markdown(
    """
    ---
    ⚠️**Disclaimer:**\n\n
    This application is for demonstration purposes only and should not be used in a production setting. The predictions are based on a trained model and may not always be accurate.\n\n
    App created by Georgios Tsamis for the scope of the course "AIDL_A02 - Neural Networks and Deep Learning" of the "MSc in Artificial Intelligence & Deep Learning" in University of West Attica.
    """
)