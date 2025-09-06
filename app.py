import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF logs

import streamlit as st
import cv2, json, random
import numpy as np

import utils  # safer than from utils import ...
get_image_files = utils.get_image_files
preprocess_image = utils.preprocess_image

import tensorflow as tf

# ---- Config (CASE-SENSITIVE on Linux/Cloud!) ----
MODEL_PATH = "Sequential2_best_model.h5"
LABEL_DECODER_PATH = "label_decoder.json"
IMAGE_DIR = "test"  # optional examples folder

st.set_page_config(page_title="Pet Detector (.h5)", layout="wide")
st.title("Pet Detector")
st.write("Cat and dog binary classification (Keras .h5).")

# ---- Helpers ----
def assert_file(path: str, min_bytes: int = 1024):
    if not os.path.exists(path):
        st.error(f"Missing file: `{path}` (check filename & location).")
        st.stop()
    size = os.path.getsize(path)
    st.sidebar.write(f"**{path}** — {size/1_000_000:.2f} MB")
    if size < min_bytes:
        st.error(f"`{path}` looks empty (size < {min_bytes} bytes). Re-upload it.")
        st.stop()

# ---- Check files early (shows sizes in sidebar) ----
st.sidebar.header("App Files")
assert_file(MODEL_PATH, 1024)          # model must be > ~1KB
assert_file(LABEL_DECODER_PATH, 10)     # labels just need to exist

# ---- Load labels ----
with open(LABEL_DECODER_PATH, "r", encoding="utf-8") as f:
    label_decoder = json.load(f)
CLASS_NAMES = [label_decoder[str(i)] for i in range(len(label_decoder))]

# ---- Load model (cached) ----
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("Model loaded")
except Exception as e:
    st.error(f"Failed to load Keras model: {e}")
    st.stop()

# ---- Choose image: uploader preferred, test/ fallback ----
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
current_image = None
caption = ""

if uploaded is not None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode the uploaded image.")
        st.stop()
    current_image = bgr
    caption = uploaded.name
else:
    image_files = get_image_files(IMAGE_DIR)
    if not image_files:
        st.info("No images in `test/`. Upload an image above to try the app.")
    else:
        # remember last selection
        if 'selected_image' not in st.session_state:
            st.session_state.selected_image = os.path.basename(image_files[0])
        selected = st.selectbox(
            "Or pick an example from `test/`",
            options=[os.path.basename(p) for p in image_files],
            index=[os.path.basename(p) for p in image_files].index(st.session_state.selected_image)
                if st.session_state.selected_image in [os.path.basename(p) for p in image_files] else 0
        )
        if selected != st.session_state.selected_image:
            st.session_state.selected_image = selected
        path = next((p for p in image_files if os.path.basename(p) == st.session_state.selected_image), None)
        if path:
            img = cv2.imread(path)
            if img is None:
                st.error(f"Could not read `{path}`.")
                st.stop()
            current_image = img
            caption = selected

# ---- Show image ----
if current_image is None or not isinstance(current_image, np.ndarray):
    st.error("Could not prepare the image. Try another file.")
    st.stop()

# Debug: confirm what we’re displaying
st.sidebar.write(
    f"img shape: {current_image.shape if current_image is not None else None}, "
    f"dtype: {current_image.dtype if current_image is not None else None}"
)

# Display (BGR)
st.image(current_image, channels="BGR", caption=caption, use_container_width=True)


# ---- Inference ----
x = preprocess_image(current_image)         # -> (1,60,80,1) float32 in [0,1]
pred = model.predict(x, verbose=0)          # (1,C) softmax OR (1,1) sigmoid
probs = pred[0]

# handle single-sigmoid binary
if probs.ndim == 0:
    probs = np.array([1. - float(probs), float(probs)], dtype=np.float32)
elif probs.shape[0] == 1 and len(CLASS_NAMES) == 2:
    p1 = float(probs[0])
    probs = np.array([1. - p1, p1], dtype=np.float32)

# softmax guard
if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
    exps = np.exp(probs - np.max(probs))
    probs = exps / np.sum(exps)

# ---- Results ----
col1, col2 = st.columns([2,1])
with col2:
    st.subheader("Probabilities")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda t: -t[1]):
        st.progress(float(p), text=f"{name}: {p:.2%}")

top_idx = int(np.argmax(probs))
st.success(f"Predicted: **{CLASS_NAMES[top_idx]}** ({probs[top_idx]:.2%})")
