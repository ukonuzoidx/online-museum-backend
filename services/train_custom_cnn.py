# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# import time


# # ✅ Suppress TensorFlow CUDA Warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # ✅ Enable GPU Memory Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


# DATASET_PATH_48 = "../data/processed/affectnet_48x48"
# batch_size=64

# num_classes = 7
# input_shape = (48, 48, 1)
# epochs = 50

# train_datagen = ImageDataGenerator(  
#     width_shift_range=0.1,           #  shifts the image horizontally by 10% of the total width                       
#     height_shift_range=0.1,          # shifts the image vertically by 10% of the total height
#     horizontal_flip=True,            # A left-facing car image might be flipped to a right-facing one
#     rescale=1./255,                  #  improving training stability , Faster Convergence
#     validation_split=0.2  
# )  


# test_datagen = ImageDataGenerator(rescale=1./255)  # Image normalization.

# train_generator = train_datagen.flow_from_directory(  
#     directory=DATASET_PATH_48,           
#     target_size=(48, 48),           
#     batch_size=32,                 
#     color_mode="grayscale",        
#     class_mode="categorical",      
#     subset="training"            
# )  

# validation_generator = train_datagen.flow_from_directory(  
#     directory=DATASET_PATH_48,  # Use DATASET_PATH_48 for validation  
#     target_size=(48, 48),          
#     batch_size=32,                 
#     color_mode="grayscale",        
#     class_mode="categorical",      
#     subset="validation"            
# )

# # test_generator = test_datagen.flow_from_directory(  
# #     directory=test_dir,  
# #     target_size=(48, 48),  
# #     batch_size=64,  
# #     color_mode="grayscale",  
# #     class_mode="categorical"  
# # )

# model = tf.keras.Sequential([
#         # input layer
#         tf.keras.layers.Input(shape=(48,48,1)),  # Input() instead of input_shape in Conv2D
#         tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu' ),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),

#         # 1st hidden dense layer
#         tf.keras.layers.Conv2D(128,(5,5), padding='same', activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),
    
#         # 2nd hidden dense layer
#         tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),
    
#         # 3rd hidden dense layer
#         tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),
    
#         # Flatten layer
#         tf.keras.layers.Flatten(),
    
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.25),
    
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.25),
        
#         # output layer
#         tf.keras.layers.Dense(7, activation='softmax')
#     ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# history = model.fit(x = train_generator,epochs = 50 ,validation_data = validation_generator)


# # save model
# model.save("custom_cnn.h5")


# # Plot training & validation accuracy
# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()



# # Class Mapping
# CLASS_MAPPING = {
#     # "neutral": 0,
#     # "happy": 1,
#     # "sad": 2,
#     # "surprise": 3,
#     # "fear": 4,
#     # "disgust": 5,
#     # "angry": 6
#     "angry": 0,
#     "disgust": 1,
#     "fear": 2,
#     "happy": 3,
#     "neutral": 4,
#     "sad": 5,
#     "surprise": 6
# }

# # Load facial data
# def load_facial_data(dataset_path):
#     data, labels = [], []
    
#     for emotion, label in CLASS_MAPPING.items():
#         emotion_path = os.path.join(dataset_path, emotion)
#         if not os.path.exists(emotion_path):
#             print(f"⚠️ Warning: Folder '{emotion}' not found in dataset path. Skipping...")
#             continue
        
#         for img_name in os.listdir(emotion_path):
#             img_path = os.path.join(emotion_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             img = img.reshape(input_shape[0], input_shape[1], 1)
#             if img is not None:
#                 data.append(img)
#                 labels.append(label)
#     return np.array(data), np.array(labels)

# x, y = load_facial_data(DATASET_PATH_48)

# # Normalize
# x = x / 255.0

# # Convert to categorical
# y = to_categorical(y, num_classes=num_classes)

# # Split data
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# # Data Augmentation
# datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
#                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# datagen.fit(x_train)

# # Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),

#     Flatten(),
#     Dense(1024, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
# model_checkpoint = ModelCheckpoint("custom_cnn.h5", save_best_only=True)

# # Train model
# history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_val, y_val), 
#                     epochs=epochs, steps_per_epoch=len(x_train) // batch_size, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# # save model
# model.save("custom_cnn.h5")

# # Plot training history
# plt.figure(figsize=(14, 7))
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.title('Training vs Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()


# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# # ✅ Suppress TensorFlow CUDA Warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # ✅ Enable GPU Memory Growth (Prevents VRAM Overload)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # ✅ Load Dataset Function
# def load_facial_data(dataset_path): 
#     data, labels = [], []
#     emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    
#     for emotion, label in emotions.items():
#         emotion_path = os.path.join(dataset_path, emotion)
#         for img_name in os.listdir(emotion_path):
#             img_path = os.path.join(emotion_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
#             if img is not None:
#                 img = cv2.resize(img, (48, 48))  # Resize to 48x48
#                 data.append(img)
#                 labels.append(label)

#     return np.array(data), np.array(labels)

# # ✅ Load and Preprocess Data
# dataset_path = "../data/processed/affectnet_48x48" 
# x, y = load_facial_data(dataset_path)

# # ✅ Normalize Data & Reshape
# x = x.reshape(-1, 48, 48, 1) / 255.0

# # ✅ Split into Train & Validation Sets (Stratified for balanced classes)
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# # ✅ Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
#     shear_range=0.2, zoom_range=0.2, horizontal_flip=True
# )
# datagen.fit(x_train)

# # ✅ Build CNN Model
# model = Sequential([
#     Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
#     BatchNormalization(),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(256, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
    
#     Flatten(),
#     Dense(512, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.4),
    
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
    
#     Dense(7, activation='softmax')
# ])

# # ✅ Compile Model
# optimizer = Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # ✅ Define Callbacks
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")

# # ✅ Train Model
# history = model.fit(
#     datagen.flow(x_train, y_train, batch_size=32),
#     validation_data=(x_val, y_val),
#     epochs=50,
#     callbacks=[reduce_lr, early_stopping, checkpoint]
# )

# # ✅ Save Final Model
# model.save("facial_emotion_model.keras")

# # ✅ Plot Training & Validation Accuracy
# plt.figure(figsize=(8, 6))
# plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s")
# plt.title("Model Accuracy Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()

# # ✅ Print Best Model
# best_epoch = np.argmax(history.history["val_accuracy"])
# best_acc = history.history["val_accuracy"][best_epoch]
# print(f"\n🏆 Best Epoch: {best_epoch+1} with Validation Accuracy: {best_acc:.4f}")


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras import regularizers
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

train_dir = "../data/raw/fer/train"
test_dir = "../data/raw/fer/test"

categories = os.listdir(train_dir)

image_counts = {category: len(os.listdir(os.path.join(train_dir, category))) for category in categories}

plt.figure(figsize=(10, 5))
sns.barplot(x=list(image_counts.keys()), y=list(image_counts.values()), palette="viridis")
plt.xlabel("Emotion Category")
plt.ylabel("Number of Images")
plt.title("Number of Images in Each Emotion Category")
plt.xticks(rotation=45)
plt.show()

categories = os.listdir(train_dir)

plt.figure(figsize=(10, 6))

for i, category in enumerate(categories):
    category_path = os.path.join(train_dir, category)
    image_path = os.path.join(category_path, os.listdir(category_path)[0])  # Get first image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    plt.subplot(2, 4, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(category)
    plt.axis('off')

plt.tight_layout()
plt.show()


train_datagen = ImageDataGenerator(  
    width_shift_range=0.1,           #  shifts the image horizontally by 10% of the total width                       
    height_shift_range=0.1,          # shifts the image vertically by 10% of the total height
    horizontal_flip=True,            # A left-facing car image might be flipped to a right-facing one
    rescale=1./255,                  #  improving training stability , Faster Convergence
    validation_split=0.2  
)  




train_generator = train_datagen.flow_from_directory(  
    directory=train_dir,           
    target_size=(48, 48),           
    batch_size=64,                 
    color_mode="grayscale",        
    class_mode="categorical",      
    subset="training"            
)  

validation_generator = train_datagen.flow_from_directory(  
    directory=train_dir,  # Use train_dir for validation  
    target_size=(48, 48),          
    batch_size=64,                 
    color_mode="grayscale",        
    class_mode="categorical",      
    subset="validation"            
)


model = tf.keras.Sequential([
        # input layer
        tf.keras.layers.Input(shape=(48,48,1)),  # Input() instead of input_shape in Conv2D
        tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu' ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),

        # 1st hidden dense layer
        tf.keras.layers.Conv2D(128,(5,5), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
    
        # 2nd hidden dense layer
        tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
    
        # 3rd hidden dense layer
        tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
    
        # Flatten layer
        tf.keras.layers.Flatten(),
    
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
    
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # output layer
        tf.keras.layers.Dense(7, activation='softmax')
    ])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()
Model: "sequential"
history = model.fit(x = train_generator,epochs = 100 ,validation_data = validation_generator)
# You can go for 100 epochs

# save model
model.save("custom_cnn.h5")

# Plot training & validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()