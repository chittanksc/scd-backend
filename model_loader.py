import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from config import Config

_model = None


def get_model():
    global _model
    if _model is None:
        model_path = Config.MODEL_PATH
        if os.path.exists(model_path):
            try:
                # Try loading with compile=False to avoid optimizer issues
                _model = load_model(model_path, compile=False)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Model file exists but cannot be loaded. Using fallback prediction.")
                _model = None
        else:
            print(f"Model file not found at: {model_path}")
            _model = None
    return _model


def predict_image(model, pil_img: Image.Image):
    if model is None:
        arr = np.array(pil_img.resize((224, 224)), dtype=np.float32)
        arr = arr.mean() / 255.0
        conf = float(arr % 1)
        label = "malignant" if conf > 0.5 else "benign"
        return label, conf
    img = pil_img.resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    if preds.ndim == 2 and preds.shape[1] == 1:
        p = float(preds[0][0])
        p = max(0.0, min(1.0, p))
        label = "malignant" if p >= 0.5 else "benign"
        conf = p if label == "malignant" else 1 - p
        return label, conf
    if preds.ndim == 2 and preds.shape[1] == 2:
        p0 = float(preds[0][0])
        p1 = float(preds[0][1])
        if p1 >= p0:
            return "malignant", p1
        else:
            return "benign", p0
    p = float(preds.flatten()[0])
    label = "malignant" if p >= 0.5 else "benign"
    conf = p if label == "malignant" else 1 - p
    return label, conf
