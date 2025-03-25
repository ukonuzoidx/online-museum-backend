
# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNetV2
# from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# import time
# from tensorflow.keras.regularizers import l2


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



# # ✅ Data Generators for CNN and Pretrained Models
# # ✅ Data Augmentation for CNN
# datagen_cnn = ImageDataGenerator(
#     rescale=1./255,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     rotation_range=30,  # ✅ Increased rotation
#     brightness_range=[0.8, 1.2],  # ✅ Adjusts brightness
#     shear_range=0.2,
#     zoom_range=0.2,
#     validation_split=0.2
# )

# # ✅ Data Augmentation for Pretrained Models
# datagen_pretrained = ImageDataGenerator(
#     rescale=1./255,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     rotation_range=20,
#     brightness_range=[0.8, 1.3],  # ✅ More contrast variation
#     shear_range=0.2,
#     zoom_range=0.2,
#     validation_split=0.2
# )


# # ✅ CNN Dataset Generator (Grayscale)
# train_generator_cnn = datagen_cnn.flow_from_directory(
#     DATASET_PATH_48,
#     target_size=img_size_cnn,
#     batch_size=batch_size,
#     color_mode="grayscale",  # ✅ Use grayscale for CNN
#     class_mode="categorical",
#     subset="training"
# )

# val_generator_cnn = datagen_cnn.flow_from_directory(
#     DATASET_PATH_48,
#     target_size=img_size_cnn,
#     batch_size=batch_size,
#     color_mode="grayscale",
#     class_mode="categorical",
#     subset="validation"
# )

# # ✅ Pretrained Model Dataset Generator (RGB)
# train_generator_pretrained = datagen_pretrained.flow_from_directory(
#     DATASET_PATH,
#     target_size=img_size_pretrained,
#     batch_size=batch_size,
#     color_mode="rgb",  # ✅ Use RGB for VGG19, ResNet50, etc.
#     class_mode="categorical",
#     subset="training"
# )

# val_generator_pretrained = datagen_pretrained.flow_from_directory(
#     DATASET_PATH,
#     target_size=img_size_pretrained,
#     batch_size=batch_size,
#     color_mode="rgb",
#     class_mode="categorical",
#     subset="validation"
# )

# # ✅ Callbacks
# early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
# checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")

# # ✅ Custom CNN Model
# def build_custom_cnn():
#     model = Sequential([
#         Input(shape=(48, 48, 1)),

#         # ✅ Add L2 Regularization to prevent overfitting
#         Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.4),

#         Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.4),

#         Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.5),

#         Flatten(),
#         Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),  # ✅ Added Regularization
#         BatchNormalization(),
#         Dropout(0.5),

#         Dense(num_classes, activation='softmax')
#     ])
    
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
#                   loss='categorical_crossentropy', metrics=['accuracy'])

#     return model



# def build_fine_tune_model(model_type):
#     """ Builds and fine-tunes pre-trained models like VGG19, ResNet50, MobileNetV2. """
#     input_layer = Input(shape=(224, 224, 3))  # ✅ Fixed Input Shape

#     # ✅ Load Pretrained Model with Proper Input Shape
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

#     # ✅ Unfreeze the last 5 layers for fine-tuning
#     for layer in base_model.layers[-5:]:  
#         layer.trainable = True  # ✅ Allow fine-tuning on the last 5 layers

#     # ✅ Add Proper Classification Head
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)  # ✅ Prevents overfitting (better than Flatten)
#     x = Dense(512, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     output = Dense(num_classes, activation="softmax")(x)

#     model = Model(inputs=base_model.input, outputs=output)

#     # ✅ Use Lower Learning Rate for Fine-Tuning
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
#                   loss="categorical_crossentropy", metrics=["accuracy"])
#     return model

# # ✅ Train & Fine-Tune Models
# models_to_train = ["Custom_CNN", "VGG19", "ResNet50", "MobileNetV2"]
# results = {}
# history_dict = {}

# for model_name in models_to_train:
#     print(f"\n🚀 Training {model_name}...\n")

#     if model_name == "Custom_CNN":
#         model = build_custom_cnn()
#         train_gen, val_gen = train_generator_cnn, val_generator_cnn
#     else:
#         model = build_fine_tune_model(model_name)
#         train_gen, val_gen = train_generator_pretrained, val_generator_pretrained

#     history = model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=epochs,
#         callbacks=[early_stop, reduce_lr, checkpoint],
#         verbose=1
#     )

#     # SAVE MODEL in ../models
#     model.save(f"../models/fine_tuned_{model_name}.keras")
#     # model.save(f"fine_tuned_{model_name}.keras")

    # history_dict[model_name] = history
    # results[model_name] = history.history["val_accuracy"][-1]

# # ✅ Best Model Selection
# best_model = max(results, key=results.get)
# print(f"\n🏆 Best Model: {best_model} with Validation Accuracy: {results[best_model]:.4f}")

# # ✅ Plot Pie Chart
# plt.figure(figsize=(6, 6))
# plt.pie(results.values(), labels=results.keys(), autopct='%1.1f%%')
# plt.title("Validation Accuracy Distribution")
# plt.show()

# # ✅ Train vs Validation Accuracy
# plt.figure(figsize=(10, 6))
# for model_name, history in history_dict.items():
#     plt.plot(history.history['accuracy'], label=f"{model_name} Train")
#     plt.plot(history.history['val_accuracy'], linestyle='dashed', label=f"{model_name} Val")
# plt.title("Training vs Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNetV2
# from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from sklearn.utils.class_weight import compute_class_weight
# import time
# from tensorflow.keras.regularizers import l2

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

# # ✅ Define Dataset Paths
# # ✅ Dataset Paths
# DATASET_PATH = "../data/processed/affectnet_224x224"  # Updated to use 224x224 dataset for everything
# batch_size = 64  
# img_size_cnn = (48, 48)
# img_size_pretrained = (224, 224)
# num_classes = 7
# epochs = 50
# # DATASET_PATH_48 = "../data/processed/affectnet_48x48"
# # DATASET_PATH_224 = "../data/processed/affectnet_224x224"
# # batch_size = 64  
# # img_size_cnn = (48, 48)
# # img_size_pretrained = (224, 224)
# # num_classes = 7
# # epochs = 50

# # ✅ Class Mapping
# CLASS_MAPPING = {
#     "neutral": 0, "happy": 1, "sad": 2, "surprise": 3, "fear": 4, "disgust": 5, "angry": 6
# }


# # ✅ Load & Resize Dataset
# def load_facial_data(dataset_path, img_size, is_grayscale=True): 
#     data, labels = [], []
#     for emotion, label in CLASS_MAPPING.items():
#         emotion_path = os.path.join(dataset_path, emotion)
#         if not os.path.exists(emotion_path):
#             print(f"⚠️ Warning: '{emotion}' folder not found. Skipping...")
#             continue

#         for img_name in tqdm(os.listdir(emotion_path), desc=f"Loading {emotion}", unit=" images"):
#             img_path = os.path.join(emotion_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
#             if img is not None:
#                 img = cv2.resize(img, img_size)  # ✅ Resize for CNN (48x48) or Pretrained Models (224x224)
#                 if is_grayscale:
#                     img = np.expand_dims(img, axis=-1)  # ✅ Ensure shape is (48,48,1) for CNN
#                 data.append(img)
#                 labels.append(label)

#     return np.array(data), np.array(labels)

# # ✅ Load and Resize Data for CNN (48x48 Grayscale)
# print("\n📥 Loading CNN Dataset (48x48)...")
# x_cnn, y = load_facial_data(DATASET_PATH, img_size_cnn, is_grayscale=True)
# x_cnn = x_cnn / 255.0  # ✅ Normalize
# y = to_categorical(y, num_classes=num_classes)

# # ✅ Train-Test Split
# x_train_cnn, x_val_cnn, y_train, y_val = train_test_split(x_cnn, y, test_size=0.2, stratify=y, random_state=42)

# # ✅ Data Augmentation for CNN
# datagen_cnn = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
#                                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# datagen_cnn.fit(x_train_cnn)

# # ✅ Pretrained Model Dataset Generator (RGB)
# datagen_pretrained = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_generator_pretrained = datagen_pretrained.flow_from_directory(
#     DATASET_PATH,
#     target_size=img_size_pretrained,
#     batch_size=batch_size,
#     color_mode="rgb",
#     class_mode="categorical",
#     subset="training"
# )

# val_generator_pretrained = datagen_pretrained.flow_from_directory(
#     DATASET_PATH,
#     target_size=img_size_pretrained,
#     batch_size=batch_size,
#     color_mode="rgb",
#     class_mode="categorical",
#     subset="validation"
# )

# # ✅ Callbacks
# early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
# checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")

# # ✅ Custom CNN Model
# def build_custom_cnn():
#     model = Sequential([
#         Input(shape=(48, 48, 1)),
#         Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.4),

#         Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.4),

#         Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.5),

#         Flatten(),
#         Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         Dropout(0.5),

#         Dense(num_classes, activation='softmax')
#     ])
    
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
#                   loss='categorical_crossentropy', metrics=['accuracy'])

#     return model

# # ✅ Pretrained Models
# def build_fine_tune_model(model_type):
#     if model_type == "VGG19":
#         base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     elif model_type == "ResNet50":
#         base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     elif model_type == "MobileNetV2":
#         base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#     else:
#         raise ValueError("Unsupported model type")

#     for layer in base_model.layers[-10:]:  
#         layer.trainable = True  

#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(512, activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     output = Dense(num_classes, activation="softmax")(x)

#     model = Model(inputs=base_model.input, outputs=output)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
#                   loss="categorical_crossentropy", metrics=["accuracy"])
#     return model

# # ✅ Train & Fine-Tune Models
# models_to_train = ["Custom_CNN", "VGG19", "ResNet50", "MobileNetV2"]
# results = {}
# history_dict = {}


# for model_name in models_to_train:
#     print(f"\n🚀 Training {model_name}...\n")
#     model = build_custom_cnn() if model_name == "Custom_CNN" else build_fine_tune_model(model_name)
#     history = model.fit(train_generator_pretrained if model_name != "Custom_CNN" else datagen_cnn.flow(x_train_cnn, y_train, batch_size=batch_size),
#                         validation_data=val_generator_pretrained if model_name != "Custom_CNN" else (x_val_cnn, y_val),
#                         epochs=epochs, callbacks=[early_stop, reduce_lr, checkpoint])

#     model.save(f"../models/fine_tuned_{model_name}.keras")
#     history_dict[model_name] = history
#     results[model_name] = history.history["val_accuracy"][-1]

# # ✅ Best Model Selection
# best_model = max(results, key=results.get)
# print(f"\n🏆 Best Model: {best_model} with Validation Accuracy: {results[best_model]:.4f}")

# # Load the best model
# model = tf.keras.models.load_model(f"../models/fine_tuned_{best_model}.keras")
# model.summary()

# # save all the models history in a txt file 
# with open("models_history.txt", "w") as file:
#     for model_name, history in history_dict.items():
#         file.write(f"{model_name}:\n")
#         for key, value in history.history.items():
#             file.write(f"{key}: {value}\n")
#         file.write("\n")
         
    
    

# # ✅ Plot Pie Chart
# plt.figure(figsize=(6, 6))
# plt.pie(results.values(), labels=results.keys(), autopct='%1.1f%%')
# plt.title("Validation Accuracy Distribution")
# plt.show()

# # ✅ Train vs Validation Accuracy
# plt.figure(figsize=(10, 6))
# for model_name, history in history_dict.items():
#     plt.plot(history.history['accuracy'], label=f"{model_name} Train")
#     plt.plot(history.history['val_accuracy'], linestyle='dashed', label=f"{model_name} Val")
# plt.title("Training vs Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()



import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import VGG19, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2

# ✅ Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ Enable GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ✅ Dataset Paths
DATASET_PATH = "../data/processed/affectnet_224x224"  # Using 224x224 dataset for both CNN and Pretrained models
batch_size = 64  
img_size_cnn = (48, 48)
img_size_pretrained = (224, 224)
num_classes = 7
epochs = 50

# ✅ Class Mapping
CLASS_MAPPING = {
    "neutral": 0, "happy": 1, "sad": 2, "surprise": 3, "fear": 4, "disgust": 5, "angry": 6
}


# ✅ Data Augmentation for CNN (48x48 Grayscale)
datagen_cnn = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=30,  # More variations
    brightness_range=[0.8, 1.2],  # Adjusts brightness
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

# ✅ CNN Dataset Generator (Grayscale)
train_generator_cnn = datagen_cnn.flow_from_directory(
    DATASET_PATH,
    target_size=img_size_cnn,
    batch_size=batch_size,
    color_mode="grayscale",  # ✅ Use grayscale for CNN
    class_mode="categorical",
    subset="training"
)

val_generator_cnn = datagen_cnn.flow_from_directory(
    DATASET_PATH,
    target_size=img_size_cnn,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# ✅ Pretrained Model Dataset Generator (RGB)
datagen_pretrained = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.3],  # More contrast variation
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

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

# ✅ Custom CNN Model (48x48 Grayscale)
def build_custom_cnn():
    model = Sequential([
        Input(shape=(48, 48, 1)),  # ✅ Grayscale Input

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

# ✅ Pretrained Models (224x224 RGB)
def build_fine_tune_model(model_type):
    if model_type == "VGG19":
        base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_type == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif model_type == "MobileNetV2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported model type")

    for layer in base_model.layers[-10:]:  
        layer.trainable = True  

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ✅ Train & Fine-Tune Models
models_to_train = ["Custom_CNN", "VGG19", "ResNet50", "MobileNetV2"]
results = {}

for model_name in models_to_train:
    print(f"\n🚀 Training {model_name}...\n")
    model = build_custom_cnn() if model_name == "Custom_CNN" else build_fine_tune_model(model_name)
    
    history = model.fit(train_generator_pretrained if model_name != "Custom_CNN" else train_generator_cnn,
                        validation_data=val_generator_pretrained if model_name != "Custom_CNN" else val_generator_cnn,
                        epochs=epochs, callbacks=[early_stop, reduce_lr, checkpoint])

    model.summary()

    model.save(f"../models/fine_tuned_{model_name}.keras")

# ✅ Done!
print("\n🎉 Training Complete!")
