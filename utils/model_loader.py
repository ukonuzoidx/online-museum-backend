import os
import tensorflow as tf
from keras.models import load_model as keras_load_model

# Local directory where models are stored
MODEL_DIR = "models"  # models are in a subdirectory

# Map model names to filenames - only VGG19
LOCAL_MODELS = {
    # "VGG19_m": "VGG19.keras",
    "VGG19": "fine_tuned_VGG19.keras"
}


def load_model(model_name: str):
    if model_name not in LOCAL_MODELS:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    filename = LOCAL_MODELS[model_name]
    model_path = os.path.join(MODEL_DIR, filename)
    # model_path = os.path.join(filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"📦 Loading model '{model_name}' from local path: {model_path}")
    model = keras_load_model(model_path)
    print(f"✅ Model '{model_name}' loaded successfully!")
    return model
