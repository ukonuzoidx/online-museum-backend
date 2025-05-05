import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
from datetime import datetime
from tensorflow.keras.applications import VGG19, ResNet50, MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Suppress TensorFlow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ✅ GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ✅ Paths
DATASET_PATH_48 = "../data/processed/affectnet_48x48"
DATASET_PATH_224 = "../data/processed/affectnet_224x224"
TEST_DATASET_PATH = "../data/test"
batch_size = 32
img_size_cnn = (48, 48)
img_size_pretrained = (224, 224)
num_classes = 7
epochs = 40
emotion_labels = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

# ✅ Results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"../results/training_run_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# ✅ Data generators
# CNN model data generator
datagen_cnn = ImageDataGenerator(
    rescale=1.0 / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)
# Training and validation data generators for CNN model
train_generator_cnn = datagen_cnn.flow_from_directory(
    DATASET_PATH_48,
    target_size=img_size_cnn,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
)
# Validation data generator for CNN model
val_generator_cnn = datagen_cnn.flow_from_directory(
    DATASET_PATH_48,
    target_size=img_size_cnn,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
)
# Pretrained model data generator
datagen_pretrained = ImageDataGenerator(
    rescale=1.0 / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)
# Training and validation data generators for pretrained model
train_generator_pretrained = datagen_pretrained.flow_from_directory(
    DATASET_PATH_224,
    target_size=img_size_pretrained,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
)
# Validation data generator for pretrained model
val_generator_pretrained = datagen_pretrained.flow_from_directory(
    DATASET_PATH_224,
    target_size=img_size_pretrained,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
)
# Test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
# Test data generator for CNN model
test_generator_cnn = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=(48, 48),
    batch_size=1,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False,
)


# ✅ Smart RGB resize generator
def rgb_and_resize_generator(generator):
    for images, labels in generator:
        images_rgb = np.repeat(images, 3, axis=-1)
        resized_images = np.array([cv2.resize(img, (224, 224)) for img in images_rgb])
        yield resized_images, labels


# ✅ Models

# Custom CNN model function
def build_custom_cnn():
    model = Sequential(
        [
            Input(shape=(48, 48, 1)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.3),
            tf.keras.layers.Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Pretrained model function
def build_fine_tune_model(model_type):
    if model_type == "VGG19":
        base_model = VGG19(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    elif model_type == "ResNet50":
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    elif model_type == "MobileNetV2":
        base_model = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    else:
        raise ValueError("Unsupported model type")
    
    
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ✅ Training
models_to_train = ["Custom_CNN", "VGG19", "ResNet50", "MobileNetV2"]
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
history_dict = {}

for model_name in models_to_train:
    print(f"Training {model_name}...")
    model = (
        build_custom_cnn()
        if model_name == "Custom_CNN"
        else build_fine_tune_model(model_name)
    )
    generator = (
        train_generator_cnn
        if model_name == "Custom_CNN"
        else train_generator_pretrained
    )
    val_generator = (
        val_generator_cnn if model_name == "Custom_CNN" else val_generator_pretrained
    )

    # Model summary and training
    model.summary()
    # fitting the model for 40 epochs
    # with early stopping and reduce learning rate callbacks
    history = model.fit(
        generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    # saving the model to the results directory
    model.save(f"{results_dir}/fine_tuned_{model_name}.keras")
    # saving the history to a dictionary
    history_dict[model_name] = history

# ✅ Evaluation
evaluation_results = {}
full_reports = {}

for model_name in models_to_train:
    print(f"Evaluating {model_name}...")
    model = tf.keras.models.load_model(f"{results_dir}/fine_tuned_{model_name}.keras")
    generator = (
        test_generator_cnn
        if model_name == "Custom_CNN"
        else rgb_and_resize_generator(test_generator_cnn)
    )
    steps = len(test_generator_cnn) if model_name != "Custom_CNN" else None
    y_pred_probs = model.predict(generator, steps=steps, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = test_generator_cnn.classes

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_confusion_matrix.png")
    plt.close()

    report = classification_report(
        y_true_classes, y_pred_classes, target_names=emotion_labels, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(
        f"{results_dir}/{model_name}_classification_report.csv"
    )
    evaluation_results[model_name] = {
        "accuracy": np.mean(y_true_classes == y_pred_classes),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }
    full_reports[model_name] = report

    print(f"\n✅ {model_name} Evaluation Done:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")

# ✅ Final Best Model
best_model = max(evaluation_results.items(), key=lambda x: x[1]["f1"])[0]
print(
    f"\n🏆 Best Model: {best_model} with F1-Score: {evaluation_results[best_model]['f1']:.4f}"
)


# ✅ Weighted F1-score comparison
plt.figure(figsize=(8, 6))
sns.barplot(
    x=list(evaluation_results.keys()), y=[v["f1"] for v in evaluation_results.values()]
)
plt.title("Weighted F1-Score Comparison")
plt.ylabel("Weighted F1-Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{results_dir}/weighted_f1_comparison.png")
plt.close()

# ✅ Emotion score comparison per model
for metric in ["precision", "recall", "f1-score"]:
    plt.figure(figsize=(10, 7))
    for model_name in models_to_train:
        scores = [full_reports[model_name][emo][metric] for emo in emotion_labels]
        plt.plot(emotion_labels, scores, marker="o", label=model_name)
    plt.title(f"Per-Emotion {metric.capitalize()} Comparison")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Emotion")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/per_emotion_{metric}_comparison.png")
    plt.close()

# ✅ pie chart of correct predictions
for model_name in models_to_train:
    model = tf.keras.models.load_model(f"{results_dir}/fine_tuned_{model_name}.keras")
    generator = (
        test_generator_cnn
        if model_name == "Custom_CNN"
        else rgb_and_resize_generator(test_generator_cnn)
    )
    steps = len(test_generator_cnn) if model_name != "Custom_CNN" else None
    y_pred_probs = model.predict(generator, steps=steps, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = test_generator_cnn.classes

    correct_preds = y_pred_classes == y_true_classes
    correct_counts = np.bincount(
        y_true_classes[correct_preds], minlength=len(emotion_labels)
    )

    plt.figure(figsize=(8, 8))
    plt.pie(correct_counts, labels=emotion_labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"Correct Predictions Distribution - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_correct_predictions_pie.png")
    plt.close()
