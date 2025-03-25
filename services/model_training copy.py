# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from torchvision import models
# import time

# # Paths to datasets
# DATASET_48_PATH = "../data/raw/affectnet_48x48"
# DATASET_224_PATH = "../data/raw/affectnet_224x224"

# # Hyperparameters
# BATCH_SIZE = 32
# EPOCHS = 20
# LEARNING_RATE = 0.001
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data transformations (48x48 and 224x224 versions)
# transform_48 = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# transform_224 = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# # Load datasets
# train_dataset_48 = datasets.ImageFolder(root=DATASET_48_PATH, transform=transform_48)
# train_loader_48 = DataLoader(train_dataset_48, batch_size=BATCH_SIZE, shuffle=True)

# train_dataset_224 = datasets.ImageFolder(root=DATASET_224_PATH, transform=transform_224)
# train_loader_224 = DataLoader(train_dataset_224, batch_size=BATCH_SIZE, shuffle=True)

# # Define model selection
# MODELS = {
#     "VGG16": models.vgg16(pretrained=True),
#     "VGG19": models.vgg19(pretrained=True),
#     "ResNet50": models.resnet50(pretrained=True),
#     "MobileNetV2": models.mobilenet_v2(pretrained=True)
# }

# # Modify the classifier for 7 emotion classes
# def modify_model(model_name, num_classes=7):
#     model = MODELS[model_name]
#     if "resnet" in model_name.lower() or "mobilenet" in model_name.lower():
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#     elif "vgg" in model_name.lower():
#         model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
#     return model.to(DEVICE)

# # Training function
# def train_model(model, train_loader, epochs=EPOCHS):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     model.train()
    
#     for epoch in range(epochs):
#         start_time = time.time()
#         running_loss = 0.0
#         correct, total = 0, 0
        
#         for images, labels in train_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_acc = 100 * correct / total
#         print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {time.time()-start_time:.2f}s")
    
#     print("Training complete!")

# # Train all models
# for model_name in MODELS.keys():
#     print(f"\nTraining {model_name} on 48x48 dataset...")
#     model_48 = modify_model(model_name)
#     train_model(model_48, train_loader_48)
    
#     print(f"\nTraining {model_name} on 224x224 dataset...")
#     model_224 = modify_model(model_name)
#     train_model(model_224, train_loader_224)
    
#     # Save trained models
#     torch.save(model_48.state_dict(), f"trained_{model_name}_48x48.pth")
#     torch.save(model_224.state_dict(), f"trained_{model_name}_224x224.pth")

# print("✅ All models trained and saved!")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from torchvision import models
# import timm  # For Vision Transformers (ViTs)
# import time
# from tqdm   import tqdm

# # Paths to datasets
# DATASET_48_PATH = "../data/processed/affectnet_48x48"
# DATASET_224_PATH = "../data/processed/affectnet_224x224"

# # Hyperparameters
# BATCH_SIZE = 64
# EPOCHS = 50
# LEARNING_RATE = 0.001
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data transformations (48x48 and 224x224 versions)
# transform_48 = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# transform_224 = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# # Load datasets
# train_dataset_48 = datasets.ImageFolder(root=DATASET_48_PATH, transform=transform_48)
# train_loader_48 = DataLoader(train_dataset_48, batch_size=BATCH_SIZE, shuffle=True)

# train_dataset_224 = datasets.ImageFolder(root=DATASET_224_PATH, transform=transform_224)
# train_loader_224 = DataLoader(train_dataset_224, batch_size=BATCH_SIZE, shuffle=True)

# # Define CNN+LSTM model
# class CNN_LSTM(nn.Module):
#     def __init__(self, num_classes=7):
#         super(CNN_LSTM, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.lstm = nn.LSTM(input_size=64*12*12, hidden_size=128, batch_first=True)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), 1, -1)
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         return x

# # Define model selection
# MODELS = {
#     "VGG16": models.vgg16(pretrained=True),
#     "VGG19": models.vgg19(pretrained=True),
#     "ResNet50": models.resnet50(pretrained=True),
#     "MobileNetV2": models.mobilenet_v2(pretrained=True),
#     "ViT": timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=7),
#     "CNN+LSTM": CNN_LSTM()
# }

# # Modify the classifier for 7 emotion classes
# def modify_model(model_name, num_classes=7):
#     model = MODELS[model_name]
#     if "resnet" in model_name.lower() or "mobilenet" in model_name.lower():
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#     elif "vgg" in model_name.lower():
#         model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
#     return model.to(DEVICE)

# # Training function
# def train_model(model, train_loader, epochs=EPOCHS):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     model.train()
    
#     from tqdm import tqdm  # Import progress bar

#     for epoch in range(epochs):
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
#         start_time = time.time()
#         running_loss = 0.0
#         correct, total = 0, 0
        
#         for images, labels in train_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_acc = 100 * correct / total
#         print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {time.time()-start_time:.2f}s")
    
#     print("Training complete!")

# # Train all models
# for model_name in MODELS.keys():
#     print(f"\nTraining {model_name} on 48x48 dataset...")
#     model_48 = modify_model(model_name)
#     train_model(model_48, train_loader_48)
    
#     print(f"\nTraining {model_name} on 224x224 dataset...")
#     model_224 = modify_model(model_name)
#     train_model(model_224, train_loader_224)
    
#     # Save trained models
#     torch.save(model_48.state_dict(), f"trained_{model_name}_48x48.pth")
#     torch.save(model_224.state_dict(), f"trained_{model_name}_224x224.pth")

# print("✅ All models trained and saved!")

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNetV2
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM, TimeDistributed, Reshape
# import timm  # For Vision Transformers (ViTs)
# import os

# # Paths to datasets
# DATASET_48_PATH = "../data/processed/affectnet_48x48"
# DATASET_224_PATH = "../data/processed/affectnet_224x224"

# # Hyperparameters
# BATCH_SIZE = 32
# EPOCHS = 50
# LEARNING_RATE = 0.001
# IMG_SIZE_48 = (48, 48)
# IMG_SIZE_224 = (224, 224)

# # Data augmentation
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# # Load datasets
# train_data_48 = train_datagen.flow_from_directory(
#     DATASET_48_PATH, target_size=IMG_SIZE_48, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
# val_data_48 = train_datagen.flow_from_directory(
#     DATASET_48_PATH, target_size=IMG_SIZE_48, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# train_data_224 = train_datagen.flow_from_directory(
#     DATASET_224_PATH, target_size=IMG_SIZE_224, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
# val_data_224 = train_datagen.flow_from_directory(
#     DATASET_224_PATH, target_size=IMG_SIZE_224, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# # Model selection
# def build_model(base_model, num_classes=7):
#     base_model.trainable = False  # Freeze the pre-trained layers
#     x = Flatten()(base_model.output)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=output)
#     return model

# # CNN + LSTM model
# def build_cnn_lstm(input_shape, num_classes=7):
#     model = Sequential([
#         TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(1,) + input_shape),
#         TimeDistributed(MaxPooling2D((2, 2))),
#         TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
#         TimeDistributed(MaxPooling2D((2, 2))),
#         TimeDistributed(Flatten()),
#         LSTM(128),
#         Dense(num_classes, activation='softmax')
#     ])
#     return model

# MODELS = {
#     "VGG16": build_model(VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE_224 + (3,))),
#     "VGG19": build_model(VGG19(weights='imagenet', include_top=False, input_shape=IMG_SIZE_224 + (3,))),
#     "ResNet50": build_model(ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE_224 + (3,))),
#     "MobileNetV2": build_model(MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE_224 + (3,))),
#     "CNN+LSTM": build_cnn_lstm(IMG_SIZE_48 + (3,))
# }

# # Training function
# def train_model(model, train_data, val_data, model_name):
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
#     model.save(f"trained_{model_name}.h5")

# # Train all models
# for model_name, model in MODELS.items():
#     print(f"\nTraining {model_name}...")
#     if model_name == "CNN+LSTM":
#         train_model(model, train_data_48, val_data_48, model_name)
#     else:
#         train_model(model, train_data_224, val_data_224, model_name)

# print("✅ All models trained and saved!")



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


# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNetV2
# from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LSTM, TimeDistributed, GlobalAveragePooling2D, Conv2D, MaxPooling2D
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# import os
# import time


# # ✅ Suppress TensorFlow CUDA Warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # ✅ Disable cuDNN Autotune (Prevents Slow First Epoch)

# # ✅ Enable GPU Memory Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # ✅ Enable Mixed Precision Training (Uses FP16 for Faster Computation)
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_bfloat16')  # ✅ Faster and more stable on RTX 3060

# # ✅ Enable XLA Compilation (Optimized Computation)
# tf.config.optimizer.set_jit(True)

# # ✅ Set Training Parameters
# DATASET_PATH = "../data/processed/affectnet_224x224"
# batch_size = 64  # ✅ Increased batch size for speed
# img_size = (224, 224)  # ✅ Use (32, 32) for ultra-fast testing
# img_size_m = [224, 224]  # ✅ Use (32, 32) for ultra-fast testing
# num_classes = 7
# epochs = 50  # ✅ Reduce epochs, use EarlyStopping to stop if no improvement

# # ✅ Data Augmentation (Helps Generalization)
# # datagen = ImageDataGenerator(
# #     rescale=1./255,
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     horizontal_flip=True,
# #     brightness_range=[0.9, 1.1],
# #     validation_split=0.2  # 20% for validation
# # )
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     width_shift_range=0.15,      # ✅ Increased shift range to generalize better
#     height_shift_range=0.15,
#     horizontal_flip=True,
#     rotation_range=30,           # ✅ Introduce random rotations
#     brightness_range=[0.8, 1.3],  # ✅ Increase contrast
#     shear_range=0.2,
#     zoom_range=0.2,
#     validation_split=0.2
# )


# # ✅ Load Data (Training & Validation)
# train_generator = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=img_size,
#     batch_size=batch_size,
#     color_mode="rgb",
#     class_mode="categorical",
#     subset="training"
# )

# val_generator = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=img_size,
#     batch_size=batch_size,
#     color_mode="rgb",
#     class_mode="categorical",
#     subset="validation"
# )

# # ✅ Callbacks (Prevents Overfitting & Saves Time)
# # ✅ Early Stopping: Stops training if no improvement after 5 epochs
# early_stop = EarlyStopping(
#     monitor="val_loss",   # Monitors validation loss
#     patience=5,           # Stops training if no improvement for 5 epochs
#     restore_best_weights=True # Loads the best weights after stopping
# )

# # ✅ Reduce Learning Rate: Lowers learning rate if validation loss stops improving
# reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# # ✅ Model Checkpoint: Saves the best model based on validation accuracy
# checkpoint = ModelCheckpoint(
#     "best_model.keras",
#     monitor="val_accuracy",
#     save_best_only=True,
#     mode="max"  # Saves only when validation accuracy improves
# )

# # ✅ Custom CNN Model (Fast Training from Scratch)
# def build_custom_cnn():
#     """ Custom CNN model for facial emotion recognition. """
#     model = Sequential([
#         tf.keras.layers.Input(shape=(48,48,1)),  # ✅ Input shape (48x48, grayscale)
#         tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),

#         tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),

#         tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.25),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.5),

#         tf.keras.layers.Dense(num_classes, activation='softmax', dtype="float32")  # ✅ Ensures mixed precision works
#     ])

#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     return model

# # ✅ Fine-Tuned Pretrained Models
# # def build_fine_tune_model(model_type):
# #     """ Builds and fine-tunes pre-trained models like VGG16, VGG19, ResNet50, MobileNetV2. """
    
# #     # ✅ Convert grayscale (1-channel) to 3-channel RGB
# #     input_layer = tf.keras.layers.Input(shape=(48, 48, 1))
# #     x = Conv2D(3, (1,1), activation="linear")(input_layer)  # ✅ Convert to 3-channel

# #     if model_type == "VGG16":
# #         base_model = VGG16(weights="imagenet", include_top=False, input_shape=(48, 48, 3))  
# #     elif model_type == "VGG19":
# #         base_model = VGG19(weights="imagenet", include_top=False, input_shape=(48, 48, 3))  
# #     elif model_type == "ResNet50":
# #         base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(48, 48, 3))  
# #     elif model_type == "MobileNetV2":
# #         base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(48, 48, 3))  
# #     else:
# #         raise ValueError("Unsupported model type")

# #     # ✅ Unfreeze last 5 layers for fine-tuning
# #     for layer in base_model.layers[:-5]:  
# #         layer.trainable = False

# #     # ✅ Add Classification Layers
# #     x = GlobalAveragePooling2D()(base_model.output)
# #     x = Dense(512, activation="relu")(x)
# #     x = BatchNormalization()(x)
# #     x = Dropout(0.5)(x)
# #     x = Dense(256, activation="relu")(x)
# #     x = BatchNormalization()(x)
# #     x = Dropout(0.3)(x)
# #     output = Dense(num_classes, activation="softmax")(x)

# #     model = Model(inputs=input_layer, outputs=output)  

# #     # ✅ Compile Model with Lower Learning Rate
# #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
# #                   loss="categorical_crossentropy",
# #                   metrics=["accuracy"])
    
# #     return model
# def build_fine_tune_model(model_type):
#     """ Builds and fine-tunes pre-trained models like VGG16, VGG19, etc. """

#     # ✅ Load Pretrained Model Without input_tensor=x (Fixes Layer Mismatch)
#     if model_type == "VGG16":
#         base_model = VGG16(weights="imagenet", include_top=False, input_shape=img_size_m + [3])
#     elif model_type == "VGG19":
#         base_model = VGG19(weights="imagenet", include_top=False, input_shape=img_size_m + [3])
#     elif model_type == "ResNet50":
#         base_model = ResNet50(weights="imagenet", include_top=False, input_shape=img_size_m + [3])
#     elif model_type == "MobileNetV2":
#         base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=img_size_m + [3])
#     else:
#         raise ValueError("Unsupported model type")

#     # ✅ Unfreeze only the last 10 layers for fine-tuning
#     for layer in base_model.layers[-10:]:  
#         layer.trainable = True

#     # ✅ Add Classification Layers
#     x = GlobalAveragePooling2D()(base_model.output)  # ✅ Fixes the input connection
#     x = Dense(512, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     output = Dense(num_classes, activation="softmax")(x)

#     model = Model(inputs=base_model.input, outputs=output)  # ✅ Fix input model connection

#     # ✅ Compile Model with Lower Learning Rate
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                   loss="categorical_crossentropy",
#                   metrics=["accuracy"])
    
#     return model

# # ✅ Train & Fine-Tune All Models
# models_to_train = ["VGG19", "ResNet50", "MobileNetV2"]
# # models_to_train = ["Custom_CNN", "VGG16", "VGG19", "ResNet50", "MobileNetV2"]
# results = {}

# for model_name in models_to_train:
#     print(f"\n🚀 Training {model_name}...\n")

#     # if model_name == "Custom_CNN":
#     #     model = build_custom_cnn()
#     # else:
#     model = build_fine_tune_model(model_name)

#     start_time = time.time()
#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=epochs,
#         callbacks=[early_stop, reduce_lr, checkpoint],
#         verbose=1
#     )
#     end_time = time.time()

#     # ✅ Save Fine-Tuned Model
#     model.save(f"fine_tuned_{model_name}.keras")

#     # ✅ Store Results
#     results[model_name] = {
#         "train_accuracy": history.history["accuracy"][-1],
#         "val_accuracy": history.history["val_accuracy"][-1],
#         "training_time": end_time - start_time
#     }

# # ✅ Select the Best Model
# best_model = max(results, key=lambda m: results[m]["val_accuracy"])
# print(f"\n🏆 Best Model: {best_model} with Validation Accuracy: {results[best_model]['val_accuracy']:.4f}")

# # ✅ Plot Pie Chart
# plt.figure(figsize=(6, 6))
# plt.pie([results[m]["val_accuracy"] for m in results], labels=list(results.keys()), autopct='%1.1f%%')
# plt.title("Validation Accuracy Distribution")
# plt.show()


# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNetV2
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

# # ✅ Define Dataset Path
# DATASET_PATH = "../data/processed/affectnet_224x224"
# DATASET_PATH_48 = "../data/processed/affectnet_48x48"
# batch_size = 64
# img_size_cnn = (48, 48)
# img_size_pretrained = (224, 224)
# num_classes = 7
# epochs = 50

# # ✅ Class Mapping
# CLASS_MAPPING = {
#     "neutral": 0, "happy": 1, "sad": 2, "surprise": 3, "fear": 4, "disgust": 5, "angry": 6
# }

# # ✅ Load Facial Data (Supports Both Sizes)
# def load_facial_data(dataset_path, img_size, is_grayscale=True): 
#     data, labels = [], []
#     for emotion, label in CLASS_MAPPING.items():
#         emotion_path = os.path.join(dataset_path, emotion)
#         for img_name in os.listdir(emotion_path):
#             if is_grayscale:
#                 img = cv2.imread(os.path.join(emotion_path, img_name), cv2.IMREAD_GRAYSCALE)
#             else:
#                img = cv2.imread(os.path.join(emotion_path, img_name), cv2.IMREAD_COLOR)
#             img = cv2.resize(img, img_size)
#             data.append(img)
#             labels.append(label)
#     return np.array(data), np.array(labels)

# # ✅ Load Dataset (48x48 for CNN, 224x224 for Pretrained Models)
# x_cnn, y = load_facial_data(DATASET_PATH_48, img_size_cnn)
# x_pretrained, _ = load_facial_data(DATASET_PATH, img_size_pretrained)

# # ✅ Normalize and Reshape Data
# x_cnn = x_cnn.reshape(-1, 48, 48, 1) / 255.0  # Normalize for CNN
# x_pretrained = x_pretrained.reshape(-1, 224, 224, 1) / 255.0  # Normalize for Pretrained Models
# y = to_categorical(y, num_classes=7)

# # ✅ Split Dataset
# x_train_cnn, x_val_cnn, y_train, y_val = train_test_split(x_cnn, y, test_size=0.2, stratify=y, random_state=42)
# x_train_pretrained, x_val_pretrained, _, _ = train_test_split(x_pretrained, y, test_size=0.2, stratify=y, random_state=42)

# # ✅ Data Augmentation
# datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
#                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# datagen.fit(x_train_cnn)

# # ✅ Callbacks
# early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
# checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")

# # ✅ Custom CNN Model
# def build_custom_cnn():
#     model = Sequential([
#         Input(shape=(48, 48, 1)),
#         Conv2D(64, (3,3), padding='same', activation='relu'),
#         BatchNormalization(),
#         Conv2D(128, (3,3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Conv2D(256, (3,3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(512, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.4),
#         Dense(256, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(7, activation='softmax')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
#                   loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # ✅ Pretrained Models (Fixed Layer Issues)
# def build_fine_tune_model(model_type):
#     input_layer = Input(shape=(224, 224, 1))
#     x = tf.image.grayscale_to_rgb(input_layer)  # ✅ Convert grayscale to RGB

#     if model_type == "VGG16":
#         base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     elif model_type == "VGG19":
#         base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     elif model_type == "ResNet50":
#         base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     elif model_type == "MobileNetV2":
#         base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     else:
#         raise ValueError("Unsupported model type")

#     for layer in base_model.layers[:-5]:  # ✅ Unfreeze last 5 layers
#         layer.trainable = False

#     # x = base_model(x, training=False)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(512, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     output = Dense(num_classes, activation="softmax")(x)

#     model = Model(inputs=base_model.input, outputs=output)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
#                   loss="categorical_crossentropy", metrics=["accuracy"])
#     return model

# # ✅ Train & Fine-Tune Models
# models_to_train = ["Custom_CNN", "VGG16", "VGG19", "ResNet50", "MobileNetV2"]
# results = {}

# for model_name in models_to_train:
#     print(f"\n🚀 Training {model_name}...\n")
    
#     if model_name == "Custom_CNN":
#         model = build_custom_cnn()
#         x_train, x_val = x_train_cnn, x_val_cnn
#     else:
#         model = build_fine_tune_model(model_name)
#         x_train, x_val = x_train_pretrained, x_val_pretrained

#     history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
#                         validation_data=(x_val, y_val), epochs=epochs,
#                         callbacks=[early_stop, reduce_lr, checkpoint], verbose=1)
#     model.save(f"fine_tuned_{model_name}.keras")

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.regularizers import l2


# ✅ Suppress TensorFlow CUDA Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ✅ Define Dataset Path
DATASET_PATH = "../data/processed/affectnet_224x224"
DATASET_PATH_48 = "../data/processed/affectnet_48x48"
batch_size = 64  
img_size_cnn = (48, 48)
img_size_pretrained = (224, 224)
num_classes = 7
epochs = 50

# ✅ Class Mapping
CLASS_MAPPING = {
    "neutral": 0, "happy": 1, "sad": 2, "surprise": 3, "fear": 4, "disgust": 5, "angry": 6
}

# # ✅ Load Facial Data with tqdm Progress Bar
# def load_facial_data(dataset_path, img_size, is_grayscale=True): 
#     data, labels = [], []
#     for emotion, label in CLASS_MAPPING.items():
#         emotion_path = os.path.join(dataset_path, emotion)
#         if not os.path.exists(emotion_path):
#             print(f"⚠️ Warning: Folder '{emotion}' not found in dataset path. Skipping...")
#             continue
        
#         files = os.listdir(emotion_path)
#         for img_name in tqdm(files, desc=f"Loading {emotion}", unit=" images"):
#             img_path = os.path.join(emotion_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
#             if img is not None:
#                 img = cv2.resize(img, img_size)
#                 data.append(img)
#                 labels.append(label)

#     return np.array(data), np.array(labels)

# # ✅ Load Dataset (48x48 for CNN, 224x224 for Pretrained Models)
# print("\n📥 Loading CNN Dataset (48x48)...")
# x_cnn, y = load_facial_data(DATASET_PATH_48, img_size_cnn)
# print("\n📥 Loading Pretrained Model Dataset (224x224)...")
# x_pretrained, _ = load_facial_data(DATASET_PATH, img_size_pretrained, is_grayscale=False)

# # ✅ Normalize and Reshape Data
# x_cnn = x_cnn.reshape(-1, 48, 48, 1) / 255.0  
# x_pretrained = x_pretrained / 255.0  
# y = to_categorical(y, num_classes=7)

# # ✅ Split Dataset
# x_train_cnn, x_val_cnn, y_train, y_val = train_test_split(x_cnn, y, test_size=0.2, stratify=y, random_state=42)
# x_train_pretrained, x_val_pretrained, _, _ = train_test_split(x_pretrained, y, test_size=0.2, stratify=y, random_state=42)

# # ✅ Data Augmentation
# datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
#                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# datagen.fit(x_train_cnn)

# ✅ Data Generators for CNN and Pretrained Models
datagen_cnn = ImageDataGenerator(rescale=1./255, validation_split=0.2)
datagen_pretrained = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ✅ CNN Dataset Generator (Grayscale)
train_generator_cnn = datagen_cnn.flow_from_directory(
    DATASET_PATH_48,
    target_size=img_size_cnn,
    batch_size=batch_size,
    color_mode="grayscale",  # ✅ Use grayscale for CNN
    class_mode="categorical",
    subset="training"
)

val_generator_cnn = datagen_cnn.flow_from_directory(
    DATASET_PATH_48,
    target_size=img_size_cnn,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# ✅ Pretrained Model Dataset Generator (RGB)
train_generator_pretrained = datagen_pretrained.flow_from_directory(
    DATASET_PATH,
    target_size=img_size_pretrained,
    batch_size=batch_size,
    color_mode="rgb",  # ✅ Use RGB for VGG19, ResNet50, etc.
    class_mode="categorical",
    subset="training"
)

val_generator_pretrained = datagen_pretrained.flow_from_directory(
    DATASET_PATH,
    target_size=img_size_pretrained,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

# ✅ Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")

# ✅ Custom CNN Model
def build_custom_cnn():
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_custom_cnn():
    model = Sequential([
        Input(shape=(48, 48, 1)),
        
        Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ✅ Pretrained Models
def build_fine_tune_model(model_type):
    input_layer = Input(shape=(224, 224, 3))
    if model_type == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_type == "VGG19":
        base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_type == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_type == "MobileNetV2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported model type")

    for layer in base_model.layers[:-5]:  
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ✅ Train & Fine-Tune Models
models_to_train = ["Custom_CNN", "VGG19", "ResNet50", "MobileNetV2"]
results = {}
history_dict = {}

for model_name in models_to_train:
    print(f"\n🚀 Training {model_name}...\n")

    if model_name == "Custom_CNN":
        model = build_custom_cnn()
        train_gen, val_gen = train_generator_cnn, val_generator_cnn
    else:
        model = build_fine_tune_model(model_name)
        train_gen, val_gen = train_generator_pretrained, val_generator_pretrained

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # SAVE MODEL in ../models
    model.save(f"../models/fine_tuned_{model_name}.keras")
    # model.save(f"fine_tuned_{model_name}.keras")

    history_dict[model_name] = history
    results[model_name] = history.history["val_accuracy"][-1]

# ✅ Best Model Selection
best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with Validation Accuracy: {results[best_model]:.4f}")

# ✅ Plot Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(results.values(), labels=results.keys(), autopct='%1.1f%%')
plt.title("Validation Accuracy Distribution")
plt.show()

# ✅ Train vs Validation Accuracy
plt.figure(figsize=(10, 6))
for model_name, history in history_dict.items():
    plt.plot(history.history['accuracy'], label=f"{model_name} Train")
    plt.plot(history.history['val_accuracy'], linestyle='dashed', label=f"{model_name} Val")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
