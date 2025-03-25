import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load Dataset Function (Supports Grayscale & RGB)
def load_facial_data(dataset_path, img_size, num_channels=1): 
    data, labels = [], []
    emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

    for emotion, label in emotions.items():
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️ Warning: Folder '{emotion}' not found in dataset path. Skipping...")
            continue

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            
            # Load images in correct format
            if num_channels == 1:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img.reshape(img_size[0], img_size[1], 1)  # Keep 1 channel
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load as RGB
            
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize to specified size
                data.append(img)
                labels.append(label)  # ✅ Ensure each image has a corresponding label

    return np.array(data), np.array(labels)

# ✅ Load Data for CNN and Deep Models
dataset_path_48 = "../data/processed/affectnet_48x48"
dataset_path_224 = "../data/processed/affectnet_224x224"

# CNN Model uses grayscale
x_48, y_48 = load_facial_data(dataset_path_48, (48, 48), num_channels=1)
x_48 = x_48 / 255.0  # Normalize

# Deep Learning Models use RGB
x_224, y_224 = load_facial_data(dataset_path_224, (224, 224), num_channels=3)
x_224 = x_224 / 255.0  # Normalize

# ✅ Split Data
x_train_48, x_val_48, y_train_48, y_val_48 = train_test_split(x_48, y_48, test_size=0.2, stratify=y_48, random_state=42)
x_train_224, x_val_224, y_train_224, y_val_224 = train_test_split(x_224, y_224, test_size=0.2, stratify=y_224, random_state=42)

# ✅ Data Augmentation
datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

datagen_224 = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

datagen.fit(x_train_48)
datagen_224.fit(x_train_224)