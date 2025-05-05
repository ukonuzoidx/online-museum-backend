import cv2
import numpy as np
import mediapipe as mp
import time
import random
import os
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox

import types

# Create a dtypes attribute in numpy module
if not hasattr(np, 'dtypes'):
    np.dtypes = types.SimpleNamespace()
    np.dtypes.StringDType = None  # This is what JAX is looking for
    
import tensorflow as tf
# ✅ Define Available Models with error handling
MODEL_PATHS = {
    "CNN": "../models/fine_tuned_Custom_CNN.keras",
    "VGG19": "../models/fine_tuned_VGG19.keras",
    "ResNet50": "../models/fine_tuned_ResNet50.keras",
    "MobileNetV2": "../models/fine_tuned_MobileNetV2.keras"
}

# Check for model availability and create a filtered list
AVAILABLE_MODELS = {}
for model_name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        AVAILABLE_MODELS[model_name] = path
    else:
        print(f"⚠️ Warning: Model {model_name} not found at {path}")

if not AVAILABLE_MODELS:
    # If no models are available in the default location, look in the current directory
    for model_name, path in MODEL_PATHS.items():
        local_path = os.path.basename(path)
        if os.path.exists(local_path):
            AVAILABLE_MODELS[model_name] = local_path

if not AVAILABLE_MODELS:
    print("❌ Error: No models found! Please check your model paths.")
    exit(1)

# ✅ Default Model Selection (pick first available model)
selected_model_name = list(AVAILABLE_MODELS.keys())[0]
model = None

# ✅ Emotion Classes
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]

# Global variables
img_size = (48, 48)  # Default
use_rgb = False  # Default

# ✅ Detect Model Input Type
def detect_input_type():
    global img_size, use_rgb
    input_shape = model.input_shape
    img_size = input_shape[1:3]  # Extract (height, width)
    color_channels = input_shape[-1]  # Extract 1 (Grayscale) or 3 (RGB)
    use_rgb = color_channels == 3
    print(f"Model input configuration: Size={img_size}, RGB={use_rgb}")

# ✅ Load model with error handling
def load_model(model_name):
    global model
    try:
        model_path = AVAILABLE_MODELS[model_name]
        print(f"🔄 Loading {model_name} model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        detect_input_type()  # Update input type for new model
        print(f"✅ Successfully loaded {model_name} model")
        return True
    except Exception as e:
        error_msg = f"❌ Error loading {model_name} model: {str(e)}"
        print(error_msg)
        messagebox.showerror("Model Error", error_msg)
        return False

# Initial model loading
load_model(selected_model_name)

# ✅ GUI for Model Selection
def switch_model(event):
    global selected_model_name
    new_model = model_var.get()
    if new_model != selected_model_name:
        selected_model_name = new_model
        success = load_model(selected_model_name)
        if success:
            status_label.config(text=f"Current model: {selected_model_name}", fg="green")
        else:
            model_var.set(selected_model_name)  # Revert to previous selection
            status_label.config(text=f"Failed to load model. Using: {selected_model_name}", fg="red")

# ✅ Create Tkinter Window (Non-blocking)
window = tk.Tk()
window.title("Emotion Recognition System")
window.geometry("400x300")
window.configure(bg="#f0f0f0")

# Style
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
style.configure("TButton", font=("Arial", 12))

header_label = tk.Label(window, text="Emotion Detection & Music Recommendation", font=("Arial", 14, "bold"), bg="#f0f0f0")
header_label.pack(pady=10)

frame = tk.Frame(window, bg="#f0f0f0")
frame.pack(fill="both", expand=True, padx=20, pady=10)

model_label = tk.Label(frame, text="Select Emotion Recognition Model:", font=("Arial", 12), bg="#f0f0f0")
model_label.pack(pady=5, anchor="w")

model_var = tk.StringVar(value=selected_model_name)
dropdown = ttk.Combobox(frame, textvariable=model_var, values=list(AVAILABLE_MODELS.keys()), state="readonly", width=30)
dropdown.pack(pady=5)
dropdown.bind("<<ComboboxSelected>>", switch_model)

status_label = tk.Label(frame, text=f"Current model: {selected_model_name}", font=("Arial", 10), fg="green", bg="#f0f0f0")
status_label.pack(pady=5)

# Session duration
duration_label = tk.Label(frame, text="Session Duration (seconds):", font=("Arial", 12), bg="#f0f0f0")
duration_label.pack(pady=5, anchor="w")

duration_var = tk.IntVar(value=40)
duration_entry = ttk.Spinbox(frame, from_=10, to=120, textvariable=duration_var, width=10)
duration_entry.pack(pady=5)

# ✅ Start Emotion Detection on Button Click
def start_session():
    if model is None:
        messagebox.showerror("Error", "No model is loaded. Please select a model first.")
        return
    
    recording_time = duration_var.get()
    print(f"\n🎥 Starting {recording_time}-Second Emotion & Body Posture Detection with {selected_model_name}...\n")
    
    # Update status
    status_label.config(text=f"Running detection with {selected_model_name}...", fg="blue")
    window.update()

    # ✅ Initialize MediaPipe Face & Pose Detector
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ✅ OpenCV Webcam Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access webcam. Please check your camera.")
        status_label.config(text=f"Current model: {selected_model_name}", fg="green")
        return

    # ✅ Set Timer
    start_time = time.time()
    emotion_predictions = []
    body_postures = []
    
    # Timer display
    elapsed_time = 0
    time_left = recording_time

    while time.time() - start_time < recording_time:
        ret, frame = cap.read()
        if not ret:
            break

        # Update timer display
        current_time = time.time()
        elapsed_time = int(current_time - start_time)
        time_left = max(0, recording_time - elapsed_time)
        
        # Add timer to frame
        cv2.putText(frame, f"Time left: {time_left}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ✅ Convert Frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Detect Faces
        face_results = face_detection.process(rgb_frame)

        # ✅ Detect Body Pose
        pose_results = pose_detector.process(rgb_frame)

        detected_emotion = "Neutral"
        body_language = "Neutral"

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Ensure coordinates are within frame
                x, y = max(0, x), max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                if w <= 0 or h <= 0:
                    continue  # Skip invalid detections

                face = frame[y:y+h, x:x+w]

                if face.shape[0] > 0 and face.shape[1] > 0:
                    try:
                        # ✅ Convert to Grayscale if Needed
                        if use_rgb:
                            face_resized = cv2.resize(face, img_size)
                        else:
                            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                            face_resized = cv2.resize(face_gray, img_size)

                        # ✅ Normalize and Reshape
                        face_normalized = face_resized / 255.0
                        face_reshaped = np.expand_dims(face_normalized, axis=0)
                        if not use_rgb:
                            face_reshaped = np.expand_dims(face_reshaped, axis=-1)

                        # ✅ Predict Emotion
                        prediction = model.predict(face_reshaped, verbose=0)
                        
                        # Check prediction shape
                        if len(prediction[0]) == len(EMOTIONS):
                            emotion_idx = np.argmax(prediction)
                            confidence = np.max(prediction)

                            detected_emotion = EMOTIONS[emotion_idx]
                            emotion_predictions.append(detected_emotion)

                            # ✅ Display Emotion
                            emotion_text = f"{detected_emotion} ({confidence*100:.1f}%)"
                            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        else:
                            print(f"⚠️ Model output mismatch: Expected {len(EMOTIONS)}, got {len(prediction[0])}")
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")

        # ✅ Body Posture Analysis
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Check if key landmarks are visible
            if mp_pose.PoseLandmark.LEFT_SHOULDER.value < len(landmarks) and \
               mp_pose.PoseLandmark.RIGHT_SHOULDER.value < len(landmarks) and \
               mp_pose.PoseLandmark.LEFT_WRIST.value < len(landmarks) and \
               mp_pose.PoseLandmark.RIGHT_WRIST.value < len(landmarks):
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # ✅ Posture Analysis
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

                if shoulder_diff > 0.05:
                    body_language = "Slouched (Uncomfortable)"
                else:
                    body_language = "Upright (Neutral)"

                # ✅ Hand Movements
                hand_above_shoulder = (left_hand.y < left_shoulder.y) or (right_hand.y < right_shoulder.y)

                if hand_above_shoulder:
                    body_language = "Excited / Surprised"
                elif (left_hand.y > left_shoulder.y + 0.2) and (right_hand.y > right_shoulder.y + 0.2):
                    body_language = "Defensive / Closed-Off"
                elif (left_hand.y > left_shoulder.y) and (right_hand.y > right_shoulder.y):
                    body_language = "Uncertain / Thinking"

                body_postures.append(body_language)
                cv2.putText(frame, body_language, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ✅ Show Webcam Feed
        cv2.imshow("Live Emotion & Body Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ✅ Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Reset status
    status_label.config(text=f"Current model: {selected_model_name}", fg="green")

    # ✅ Determine Results
    final_emotion = Counter(emotion_predictions).most_common(1)[0][0] if emotion_predictions else "Neutral"
    final_body_posture = Counter(body_postures).most_common(1)[0][0] if body_postures else "Neutral"

    # ✅ Show Results in UI
    result_window = tk.Toplevel(window)
    result_window.title("Analysis Results")
    result_window.geometry("400x300")
    result_window.configure(bg="#f5f5f5")
    
    tk.Label(result_window, text="Your Emotion Analysis Results", font=("Arial", 14, "bold"), bg="#f5f5f5").pack(pady=10)
    
    results_frame = tk.Frame(result_window, bg="#f5f5f5")
    results_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    tk.Label(results_frame, text=f"Detected Emotion: {final_emotion}", font=("Arial", 12), bg="#f5f5f5").pack(anchor="w", pady=5)
    tk.Label(results_frame, text=f"Body Posture: {final_body_posture}", font=("Arial", 12), bg="#f5f5f5").pack(anchor="w", pady=5)
    
    # ✅ 🎶 Music Recommendations
    music_genres = {
        "Happy": ["Pop", "Dance", "Upbeat Jazz"],
        "Sad": ["Soft Piano", "Acoustic", "Lo-Fi"],
        "Angry": ["Rock", "Heavy Metal", "Hard Hip-Hop"],
        "Surprise": ["Electronic", "Fast Jazz", "Funky"],
        "Fear": ["Chill Lo-Fi", "Classical", "Ambient"],
        "Disgust": ["Smooth Jazz", "R&B", "Soul"],
        "Neutral": ["Indie", "Soft Rock", "Chill Pop"]
    }

    if final_body_posture in ["Slouched (Uncomfortable)", "Defensive / Closed-Off"]:
        selected_genre = "Chill Lo-Fi"
    else:
        selected_genre = random.choice(music_genres.get(final_emotion, ["Indie", "Chill Pop"]))

    tk.Label(results_frame, text="🎵 Music Recommendations:", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(anchor="w", pady=10)
    tk.Label(results_frame, text=f"Recommended Genre: {selected_genre}", font=("Arial", 12), bg="#f5f5f5").pack(anchor="w", pady=5)
    
    # Close button
    ttk.Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)

    # Print to console too
    print(f"\n✅ Final Detected Emotion: {final_emotion}")
    print(f"✅ Final Body Posture: {final_body_posture}")
    print(f"🎵 Recommended Music Genre: {selected_genre}")

# ✅ Button to Start Session
start_button = tk.Button(
    frame,
    text="Start Emotion Detection Session", 
    command=start_session,
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5,
    cursor="hand2"
)
start_button.pack(pady=15)

# Footer
footer = tk.Label(window, text="Press 'q' to exit the camera view", font=("Arial", 9), fg="gray", bg="#f0f0f0")
footer.pack(pady=5)

# ✅ Keep GUI Open
window.mainloop()
