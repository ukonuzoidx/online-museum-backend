import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Load Dataset
DATASET_PATH = "../data/"
CSV_PATH = os.path.join(DATASET_PATH, "AffectNet-C2A2.csv")
IMAGE_PATH = os.path.join(DATASET_PATH, "images")  # Change if needed

df = pd.read_csv(CSV_PATH)
print(df.head())

# ✅ Handle Missing Values
df.dropna(inplace=True)

# ✅ Visualize Data Distribution
plt.figure(figsize=(12, 5))
sns.histplot(df["valence"], bins=50, kde=True, label="Valence")
sns.histplot(df["arousal"], bins=50, kde=True, color="red", label="Arousal")
plt.legend()
plt.title("Valence & Arousal Distribution")
plt.show()

# ✅ Load & Preprocess Images
def load_image(image_name):
    img_path = os.path.join(IMAGE_PATH, image_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
    return img

df["image_data"] = df["image"].apply(load_image)

# ✅ Remove Rows with Unreadable Images
df = df.dropna(subset=["image_data"])

# ✅ Convert to NumPy Arrays
X = np.stack(df["image_data"].values)  # Images
y = df[["valence", "arousal"]].values  # Labels

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build Model
def build_model():
    model = Sequential([
        Flatten(input_shape=(224, 224, 3)),
        Dense(1024, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(2, activation="linear")  # Predicts valence & arousal
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
    return model

model = build_model()

# ✅ Train Model
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr]
)

# ✅ Save Model
model.save("affectnet_model.keras")

# ✅ Plot Training History
plt.figure(figsize=(10, 5))
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE", linestyle="dashed")
plt.legend()
plt.title("Training Progress")
plt.show()
