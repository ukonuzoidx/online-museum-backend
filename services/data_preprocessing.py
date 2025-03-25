# import os
# import shutil
# import cv2
# from pathlib import Path
# from tqdm import tqdm  # Progress bar

# # Define paths
# source_dir = "../data/raw/affectnet"  # Update this with your actual dataset path
# dest_dir = "../data/processed"  # Destination for cleaned and structured data

# # Class mapping
# class_mapping = {
#     "class001": "neutral",
#     "class002": "happy",
#     "class003": "sad",
#     "class004": "surprise",
#     "class005": "fear",
#     "class006": "angry",
#     "class007": "disgust"
# }

# # # Ensure destination directory exists
# # dest_dir.mkdir(parents=True, exist_ok=True)

# # Create folders for each emotion class
# for emotion in class_mapping.values():
#     (dest_dir / Path(emotion)).mkdir(parents=True, exist_ok=True)

# # Get all image files
# image_files = list(source_dir.glob("*.jpg"))
# print(f"Found {len(image_files)} images. Processing...\n")

# # Process images with a progress bar
# for img_path in tqdm(image_files, desc="Processing Images"):
#     try:
#         filename = img_path.name

#         # Extract class from filename (handles variations)
#         class_code = [part for part in filename.split("_") if "class" in part]
#         if not class_code:
#             print(f"Skipping (No class found): {filename}")
#             continue
        
#         class_code = class_code[0]  # Extract classXXX
#         emotion = class_mapping.get(class_code, None)
#         if not emotion:
#             print(f"Skipping (Unknown class): {filename}")
#             continue

#         # Load and check image
#         img = cv2.imread(str(img_path))
#         if img is None:
#             print(f"Skipping (Unreadable file): {filename}")
#             continue

#         # Convert to grayscale & resize
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_resized = cv2.resize(img_gray, (48, 48))

#         # Save to structured dataset
#         save_path = dest_dir / Path(emotion) / filename
#         cv2.imwrite(str(save_path), img_resized)

#     except Exception as e:
#         print(f"Error processing {filename}: {e}")

# print("\n✅ Data cleaning & restructuring completed successfully!")

# import os
# import re
# import shutil
# import cv2
# from pathlib import Path
# from tqdm import tqdm  # Progress bar

# # Define paths (UPDATE these to your actual dataset paths)
# RAW_DATASET_PATH = Path("../data/raw/affectnet")  # Change to your actual dataset path
# SORTED_DATASET_PATH = Path("../data/processed")  # Destination for organized dataset
# LOG_FILE = "removed_images.log"  # Log file for removed images

# # Define the class-to-emotion mapping
# CLASS_MAPPING = {
#     "class001": "neutral",
#     "class002": "happy",
#     "class003": "sad",
#     "class004": "surprise",
#     "class005": "fear",
#     "class006": "disgust",
#     "class007": "angry"
# }

# # Ensure output folders exist
# for emotion in CLASS_MAPPING.values():
#     (SORTED_DATASET_PATH / emotion).mkdir(parents=True, exist_ok=True)

# # Track progress per emotion
# progress = {emotion: 0 for emotion in CLASS_MAPPING.values()}
# removed_files = []

# # Load OpenCV face detector (pre-trained model)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Get image files
# image_files = [f for f in RAW_DATASET_PATH.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
# print(f"Found {len(image_files)} images. Processing...\n")

# # Process and move images
# for index, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
#     try:
#         # Extract class from filename
#         match = re.search(r'class00[1-7]', img_path.name)
#         if not match:
#             removed_files.append(f"No class found: {img_path.name}")
#             continue

#         class_label = match.group()
#         emotion = CLASS_MAPPING.get(class_label, "unknown")

#         if emotion == "unknown":
#             removed_files.append(f"Unknown class: {img_path.name}")
#             continue

#         # Load image
#         img = cv2.imread(str(img_path))
#         if img is None:
#             removed_files.append(f"Corrupt image: {img_path.name}")
#             continue

#         # Convert to grayscale for face detection
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Detect faces (ensures it's a face image)
#         faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         if len(faces) == 0:
#             removed_files.append(f"No face detected: {img_path.name}")
#             continue

#         # Resize & save image
#         img_resized = cv2.resize(img_gray, (48, 48))
#         new_filename = f"{emotion}_{progress[emotion]+1:05d}.jpg"
#         dest_path = SORTED_DATASET_PATH / emotion / new_filename

#         # Move file
#         cv2.imwrite(str(dest_path), img_resized)
#         progress[emotion] += 1

#     except Exception as e:
#         removed_files.append(f"Error processing {img_path.name}: {e}")

# # Write removed images to a log file
# with open(LOG_FILE, "w") as log:
#     for entry in removed_files:
#         log.write(entry + "\n")

# print("\n✅ Sorting & Cleanup Complete!")
# print(f"Removed {len(removed_files)} invalid images. See {LOG_FILE} for details.")
# print("\nFinal Breakdown:")
# for emotion, count in progress.items():
#     print(f"{emotion}: {count} images")


import os
import re
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Define dataset paths (UPDATE these)
RAW_DATASET_PATH = Path("../data/raw/affectnet")  # Path to unprocessed dataset
SORTED_48_PATH = Path("../data/processed/affectnet_48x48")  # Folder for 48x48 dataset
SORTED_224_PATH = Path("../data/processed/affectnet_224x224")  # Folder for 224x224 dataset
LOG_FILE = "removed_images.log"  # Log file for removed images

# Define the class-to-emotion mapping
CLASS_MAPPING = {
    "class001": "neutral",
    "class002": "happy",
    "class003": "sad",
    "class004": "surprise",
    "class005": "fear",
    "class006": "disgust",
    "class007": "angry"
}

# Ensure output folders exist for both resolutions
for emotion in CLASS_MAPPING.values():
    (SORTED_48_PATH / emotion).mkdir(parents=True, exist_ok=True)
    (SORTED_224_PATH / emotion).mkdir(parents=True, exist_ok=True)

# Track progress per emotion
progress_48 = {emotion: 0 for emotion in CLASS_MAPPING.values()}
progress_224 = {emotion: 0 for emotion in CLASS_MAPPING.values()}
removed_files = []

# Load OpenCV face detector (pre-trained model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get all image files
image_files = [f for f in RAW_DATASET_PATH.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
print(f"Found {len(image_files)} images. Processing...\n")

# Process and move images
for index, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
    try:
        # Extract class from filename
        match = re.search(r'class00[1-7]', img_path.name)
        if not match:
            removed_files.append(f"No class found: {img_path.name}")
            continue

        class_label = match.group()
        emotion = CLASS_MAPPING.get(class_label, "unknown")

        if emotion == "unknown":
            removed_files.append(f"Unknown class: {img_path.name}")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            removed_files.append(f"Corrupt image: {img_path.name}")
            continue

        # Convert to grayscale for face detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.IMREAD_COLOR)

        # Detect faces (ensures it's a face image)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            removed_files.append(f"No face detected: {img_path.name}")
            continue

        # Resize to both resolutions
        img_resized_48 = cv2.resize(img_gray, (48, 48))
        img_resized_224 = cv2.resize(img_rgb, (224, 224))

        # Save images in both datasets
        new_filename_48 = f"{emotion}_{progress_48[emotion]+1:05d}.jpg"
        new_filename_224 = f"{emotion}_{progress_224[emotion]+1:05d}.jpg"
        
        dest_path_48 = SORTED_48_PATH / emotion / new_filename_48
        dest_path_224 = SORTED_224_PATH / emotion / new_filename_224

        cv2.imwrite(str(dest_path_48), img_resized_48)
        cv2.imwrite(str(dest_path_224), img_resized_224)

        progress_48[emotion] += 1
        progress_224[emotion] += 1

    except Exception as e:
        removed_files.append(f"Error processing {img_path.name}: {e}")

# Write removed images to a log file
with open(LOG_FILE, "w") as log:
    for entry in removed_files:
        log.write(entry + "\n")

print("\n✅ Sorting & Cleanup Complete!")
print(f"Removed {len(removed_files)} invalid images. See {LOG_FILE} for details.")
print("\nFinal Breakdown:")
for emotion, count in progress_48.items():
    print(f"{emotion}: {count} images (48x48), {progress_224[emotion]} images (224x224)")

DATASET_PATH = "../data/processed/affectnet_48x48"
# Get class distribution
class_counts = {}
for emotion in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, emotion)
    if os.path.isdir(folder_path):
        class_counts[emotion] = len(os.listdir(folder_path))

# Convert to sorted dictionary
class_counts = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))

# Plot Class Distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
plt.xlabel("Emotion Classes")
plt.ylabel("Number of Images")
plt.title("Number of Images in Each Emotion Category")
plt.xticks(rotation=45)
plt.show()

# Sample Images from Each Class
plt.figure(figsize=(12, 8))
for i, emotion in enumerate(class_counts.keys()):
    folder_path = os.path.join(DATASET_PATH, emotion)
    image_file = random.choice(os.listdir(folder_path))
    image_path = os.path.join(folder_path, image_file)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(emotion)
    plt.axis("off")

plt.suptitle("Sample Images from Each Emotion Class", fontsize=16)
plt.show()

