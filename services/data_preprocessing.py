import os
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN

# Define dataset paths
RAW_DATASET_PATH = Path("../data/raw/affectnet")
SORTED_48_PATH = Path("../data/processed/affectnet_48x48")
SORTED_224_PATH = Path("../data/processed/affectnet_224x224")
LOG_FILE = "removed_images.log"

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

# Create output directories
for emotion in CLASS_MAPPING.values():
    (SORTED_48_PATH / emotion).mkdir(parents=True, exist_ok=True)
    (SORTED_224_PATH / emotion).mkdir(parents=True, exist_ok=True)

# Track progress per emotion
progress_48 = {emotion: 0 for emotion in CLASS_MAPPING.values()}
progress_224 = {emotion: 0 for emotion in CLASS_MAPPING.values()}
removed_files = []

# Initialize MTCNN detector
detector = MTCNN()

# Get all image files
image_files = [f for f in RAW_DATASET_PATH.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
print(f"Found {len(image_files)} images. Processing...\n")

# Set maximum images per class (for balanced dataset)
MAX_IMAGES_PER_CLASS = 5000  # Adjust as needed

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
            
        # Check if we've reached max images for this class
        if progress_48[emotion] >= MAX_IMAGES_PER_CLASS:
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            removed_files.append(f"Corrupt image: {img_path.name}")
            continue

        # Convert to RGB for MTCNN (it expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces with MTCNN
        faces = detector.detect_faces(img_rgb)
        
        if not faces or len(faces) == 0:
            removed_files.append(f"No face detected: {img_path.name}")
            continue
        
        # Get the face with the highest confidence
        face = max(faces, key=lambda x: x['confidence'])
        
        # Only use faces with good confidence
        if face['confidence'] < 0.9:  # Adjust threshold as needed
            removed_files.append(f"Low confidence face: {img_path.name}")
            continue
            
        # Extract face coordinates
        x, y, width, height = face['box']
        
        # Add margin (20%)
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        
        # Adjusted coordinates with margin
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        width = min(img.shape[1] - x, width + 2*margin_x)
        height = min(img.shape[0] - y, height + 2*margin_y)
        
        # Crop face region
        face_img = img_rgb[y:y+height, x:x+width]
        
        # Convert RGB to grayscale for 48x48
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        
        # Resize images
        face_48 = cv2.resize(face_gray, (48, 48))
        face_224 = cv2.resize(face_img, (224, 224))
        
        # Save images
        new_filename_48 = f"{emotion}_{progress_48[emotion]+1:05d}.jpg"
        new_filename_224 = f"{emotion}_{progress_224[emotion]+1:05d}.jpg"
        
        dest_path_48 = SORTED_48_PATH / emotion / new_filename_48
        dest_path_224 = SORTED_224_PATH / emotion / new_filename_224
        
        # Convert back to BGR for OpenCV writing
        face_224_bgr = cv2.cvtColor(face_224, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(dest_path_48), face_48)
        cv2.imwrite(str(dest_path_224), face_224_bgr)
        
        progress_48[emotion] += 1
        progress_224[emotion] += 1

    except Exception as e:
        removed_files.append(f"Error processing {img_path.name}: {str(e)}")

# Write removed images to log file
with open(LOG_FILE, "w") as log:
    for entry in removed_files:
        log.write(entry + "\n")

# Print summary
print("\n✅ Preprocessing Complete!")
print(f"Removed {len(removed_files)} invalid images. See {LOG_FILE} for details.")
print("\nFinal Class Distribution:")
for emotion, count in progress_48.items():
    print(f"{emotion}: {count} images")

# Calculate class weights for training (to handle imbalance)
total_images = sum(progress_48.values())
class_weights = {i: total_images / (len(CLASS_MAPPING) * count) 
                for i, (emotion, count) in enumerate(progress_48.items())}

print("\nClass Weights for Training (to handle class imbalance):")
for idx, (emotion, weight) in enumerate(zip(progress_48.keys(), class_weights.values())):
    print(f"{emotion} (class {idx}): {weight:.2f}")

# Save class weights to file for training
import json
with open("class_weights.json", "w") as f:
    json.dump(class_weights, f)
