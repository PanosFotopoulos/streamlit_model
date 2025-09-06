import os
import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def get_image_files(directory: str):
    if not os.path.isdir(directory):
        return []
    files = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(directory, fname))
    return sorted(files)

def preprocess_image(img):
    """
    Accepts either:
      - a NumPy BGR image (as loaded by cv2.imread), or
      - a string path to an image file
    Returns: array of shape (1, 60, 80, 1) float32 in [0,1]
    """
    if isinstance(img, str):
        img = cv2.imread(img)  # BGR
    if img is None:
        raise ValueError("preprocess_image: invalid image input (None)")

    img = cv2.resize(img, (80, 60))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, :, :, np.newaxis]  # (1, 60, 80, 1)
    return img

__all__ = ["get_image_files", "preprocess_image"]
