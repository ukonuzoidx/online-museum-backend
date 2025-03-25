import tensorflow as tf
import os

# # ✅ Define Model Paths
# MODEL_PATHS = {
#     "CNN": "backend/models/fine_tuned_Custom_CNN.keras",
#     "VGG19": "backend/models/fine_tuned_VGG19.keras",
#     "ResNet50": "backend/models/fine_tuned_ResNet50.keras",
#     "MobileNetV2": "backend/models/fine_tuned_MobileNetV2.keras"
# }

# def load_model(model_name="CNN"):
#     """Loads a trained emotion recognition model."""
#     if model_name not in MODEL_PATHS:
#         raise ValueError(f"❌ Model '{model_name}' not found! Choose from: {list(MODEL_PATHS.keys())}")
    
#     print(f"🔄 Loading model: {model_name} ...")
#     model = tf.keras.models.load_model(MODEL_PATHS[model_name])
#     print(f"✅ {model_name} model loaded successfully!")
#     return model


import os
import gdown
from keras.models import load_model as keras_load_model

# Directory to store downloaded models
MODEL_DIR = "models"

# Map model names to their Google Drive file IDs
MODEL_URLS = {
    "CNN": "1SBC_rl-AFFz25_0jz1S9tFLqxRzvKArC",  
    "VGG19": "1O5IzQ1UyWa4cgzIlR1yQnoTmFWQwELTB",   
    "ResNet50": "1QQuF16x5mN5T3P422ZPtUhtxduD4DF3E",
    "MobileNetV2": "1ZZWBrw6aA-XfhMMNq9FQAZM7RJTWwWn7"
}

def download_model_from_drive(file_id, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    file_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(file_path):
        print(f"⬇️ Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)

    return file_path

def load_model(model_name: str):
    file_id = MODEL_URLS.get(model_name)
    if not file_id:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    filename = f"{model_name}.keras"
    model_path = download_model_from_drive(file_id, filename)
    model = keras_load_model(model_path)
    print(f"✅ Model '{model_name}' loaded from: {model_path}")
    return model
