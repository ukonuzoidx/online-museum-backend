# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # import mediapipe as mp


# # # ✅ Load Model
# # MODEL_PATH = "facial_emotion_model.keras"  # Change to "fine_tuned_VGG19.keras" for others
# # # MODEL_PATH = "../models/fine_tuned_VGG19.keras"  # Change to "fine_tuned_VGG19.keras" for others
# # model = load_model(MODEL_PATH)

# # # ✅ Emotion Labels
# # EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]

# # # OpenCV Face Detector
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # MediaPipe Pose Detector (For Body Language)
# # mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose()


# # # ✅ Define Model Input Shape
# # input_size = (48, 48) 
# # # input_size = (48, 48) if "Custom_CNN" in MODEL_PATH else (224, 224)
# # is_grayscale = "Custom_CNN" in MODEL_PATH  # Custom CNN expects grayscale images


# # # ✅ Start Webcam
# # cap = cv2.VideoCapture(1)

# # print("📸 Press 'q' to exit.")

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # Convert to grayscale for face detection
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# #     # Detect faces
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

# #     for (x, y, w, h) in faces:
# #         face = frame[y:y+h, x:x+w]  # Crop face

# #         # Convert to grayscale if needed
# #         # if is_grayscale:
# #         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #         # Resize image to match model input
# #         face_img = cv2.resize(face, input_size)
        
# #         # Ensure correct shape
# #         # if is_grayscale:
# #         face_img = np.expand_dims(face_img, axis=-1)  # Keep single channel
        
# #         face_img = np.expand_dims(face_img, axis=0)  # Expand batch dimension
# #         face_img = face_img / 255.0  # Normalize

# #         # Predict Emotion
# #         predictions = model.predict(face_img)[0]
# #         emotion_index = np.argmax(predictions)
# #         emotion_label = EMOTIONS[emotion_index]
# #         confidence = predictions[emotion_index] * 100

# #         # Overlay text on frame
# #         text = f"{emotion_label} ({confidence:.2f}%)"
# #         cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# #     # Display frame
# #     cv2.imshow("Emotion Detection", frame)

# #     # Press 'q' to quit
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release capture and close windows
# # cap.release()
# # cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp

# # ✅ Load Trained Model (VGG19 Fine-Tuned)
# model_path = "../models/fine_tuned_MobileNetV2.keras"  # Adjust path if necessary
# model = tf.keras.models.load_model(model_path)

# # ✅ Emotion Labels
# EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

# # ✅ Initialize OpenCV Webcam
# cap = cv2.VideoCapture(0)

# # ✅ Initialize MediaPipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# # ✅ Real-Time Loop
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert frame to RGB (MediaPipe requirement)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_frame)
    
#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
#             # Ensure bounding box is within frame
#             if x >= 0 and y >= 0 and x + w <= iw and y + h <= ih:
#                 face = frame[y:y+h, x:x+w]
#                 face = cv2.resize(face, (224, 224))  # Resize for VGG19
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
#                 face = np.expand_dims(face, axis=0)
                
#                 # Predict Emotion
#                 preds = model.predict(face)[0]
#                 emotion_idx = np.argmax(preds)
#                 emotion = EMOTIONS[emotion_idx]
#                 confidence = preds[emotion_idx] * 100
                
#                 # Display Results
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y-10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
#     # Show Frame
#     cv2.imshow("Real-Time Emotion Detection", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import time
# from collections import Counter

# # ✅ Load the Best Model (Change Model Path Accordingly)
# MODEL_PATH = "custom_cnn.h5"  # Change as needed
# # MODEL_PATH = "../models/fine_tuned_Custom_CNN.keras"  # Change as needed
# model = tf.keras.models.load_model(MODEL_PATH)

# # ✅ Define Emotion Classes
# EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]

# # ✅ Detect Model Input Type (Grayscale or RGB)
# input_shape = model.input_shape
# img_size = input_shape[1:3]  # Extracts (height, width)
# color_channels = input_shape[-1]  # Extracts 1 (Grayscale) or 3 (RGB)
# use_rgb = color_channels == 3  # ✅ True if model needs RGB

# # ✅ Initialize MediaPipe Face Detector
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# # ✅ OpenCV Webcam Capture
# cap = cv2.VideoCapture(0)

# # ✅ Parameters
# recording_time = 70  # Seconds to record expressions
# frame_rate = 10  # How many frames to capture per second
# frames_to_capture = recording_time * frame_rate
# emotion_predictions = []  # Stores detected emotions

# print(f"\n🎥 Recording Facial Expressions for {recording_time} Seconds...")

# start_time = time.time()

# while time.time() - start_time < recording_time:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # ✅ Convert Frame to RGB for MediaPipe Detection
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # ✅ Detect Faces
#     results = face_detection.process(rgb_frame)

#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

#             # ✅ Extract Face Region
#             face = frame[y:y+h, x:x+w]

#             if face.shape[0] > 0 and face.shape[1] > 0:
#                 # ✅ Convert to Grayscale if Needed
#                 if use_rgb:
#                     face_resized = cv2.resize(face, img_size)  # Keep RGB format
#                 else:
#                     face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
#                     face_resized = cv2.resize(face_gray, img_size)  # Resize

#                 # ✅ Normalize and Reshape
#                 face_normalized = face_resized / 255.0
#                 face_reshaped = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
#                 if not use_rgb:
#                     face_reshaped = np.expand_dims(face_reshaped, axis=-1)  # Add channel for CNN

#                 # ✅ Predict Emotion
#                 prediction = model.predict(face_reshaped, verbose=0)
#                 emotion_idx = np.argmax(prediction)
#                 confidence = np.max(prediction)

#                 # ✅ Store Prediction
#                 emotion_predictions.append(EMOTIONS[emotion_idx])

#                 # ✅ Display Live Feedback
#                 emotion_text = f"{EMOTIONS[emotion_idx]} ({confidence*100:.1f}%)"
#                 cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # ✅ Show Webcam Feed
#     cv2.imshow("Recording Facial Expression...", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ✅ Analyze Final Emotion Result
# cap.release()
# cv2.destroyAllWindows()

# # ✅ Determine Most Frequent Emotion
# final_emotion = Counter(emotion_predictions).most_common(1)[0][0]
# print(f"\n✅ Final Detected Emotion: {final_emotion}")

# # ✅ TODO: Map Emotion to Music Selection Logic 🎵
# if final_emotion == "Happy":
#     print("🎶 Playing Uplifting Music!")
# elif final_emotion == "Sad":
#     print("🎶 Playing Comforting Music!")
# elif final_emotion == "Angry":
#     print("🎶 Playing Calm Music!")
# elif final_emotion == "Surprise":
#     print("🎶 Playing Exciting Music!")
# elif final_emotion == "Fear":
#     print("🎶 Playing Soothing Music!")
# elif final_emotion == "Disgust":
#     print("🎶 Playing Neutral Music!")
# else:
#     print("🎶 Playing Balanced Music!")


# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import time
# from collections import Counter
# import random

# # ✅ Load the Best Model
# MODEL_PATH = "../models/fine_tuned_VGG19.keras"  # Change for different models
# model = tf.keras.models.load_model(MODEL_PATH)

# # ✅ Define Emotion Classes
# EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Disgust", "Angry"]

# # ✅ Detect Model Input Type (Grayscale or RGB)
# input_shape = model.input_shape
# img_size = input_shape[1:3]  # Extracts (height, width)
# color_channels = input_shape[-1]  # Extracts 1 (Grayscale) or 3 (RGB)
# use_rgb = color_channels == 3  # ✅ True if model needs RGB

# # ✅ Initialize MediaPipe Face & Pose Detector
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# mp_pose = mp.solutions.pose
# pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # ✅ OpenCV Webcam Capture
# cap = cv2.VideoCapture(0)

# # ✅ User Customization: Set Recording Time
# RECORDING_TIME = 40  # ⏳ Set recording duration in seconds
# FRAME_RATE = 10  # 🎥 Number of frames per second
# frames_to_capture = RECORDING_TIME * FRAME_RATE
# emotion_predictions = []  # Stores detected emotions
# body_postures = []  # Stores detected body postures

# print(f"\n🎥 Recording Facial & Body Expressions for {RECORDING_TIME} Seconds...")

# start_time = time.time()

# while time.time() - start_time < RECORDING_TIME:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # ✅ Convert Frame to RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # ✅ Detect Faces
#     face_results = face_detection.process(rgb_frame)

#     # ✅ Detect Body Pose
#     pose_results = pose_detector.process(rgb_frame)

#     detected_emotion = None
#     body_language = "Neutral"

#     if face_results.detections:
#         for detection in face_results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

#             # ✅ Extract Face
#             face = frame[y:y+h, x:x+w]

#             if face.shape[0] > 0 and face.shape[1] > 0:
#                 # ✅ Convert to Grayscale if Needed
#                 if use_rgb:
#                     face_resized = cv2.resize(face, img_size)
#                 else:
#                     face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                     face_resized = cv2.resize(face_gray, img_size)

#                 # ✅ Normalize and Reshape
#                 face_normalized = face_resized / 255.0
#                 face_reshaped = np.expand_dims(face_normalized, axis=0)
#                 if not use_rgb:
#                     face_reshaped = np.expand_dims(face_reshaped, axis=-1)

#                 # ✅ Predict Emotion
#                 prediction = model.predict(face_reshaped, verbose=0)
#                 emotion_idx = np.argmax(prediction)
#                 confidence = np.max(prediction)

#                 # ✅ Store Prediction
#                 detected_emotion = EMOTIONS[emotion_idx]
#                 emotion_predictions.append(detected_emotion)

#                 # ✅ Display Emotion in Frame
#                 emotion_text = f"{detected_emotion} ({confidence*100:.1f}%)"
#                 cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # ✅ Improved Body Language Detection
#     if pose_results.pose_landmarks:
#         landmarks = pose_results.pose_landmarks.landmark

#         left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
#         right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

#         # ✅ Posture Analysis
#         shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

#         if shoulder_diff > 0.05:
#             body_language = "Slouched (Uncomfortable)"
#         else:
#             body_language = "Upright (Neutral)"

#         # ✅ Hand Movements
#         hand_above_shoulder = (left_hand.y < left_shoulder.y) or (right_hand.y < right_shoulder.y)

#         if hand_above_shoulder:
#             body_language = "Excited / Surprised"
#         elif (left_hand.y > left_shoulder.y + 0.2) and (right_hand.y > right_shoulder.y + 0.2):
#             body_language = "Defensive / Closed-Off"
#         elif (left_hand.y > left_shoulder.y) and (right_hand.y > right_shoulder.y):
#             body_language = "Uncertain / Thinking"

#         body_postures.append(body_language)
#         cv2.putText(frame, body_language, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#     # ✅ Show Webcam Feed
#     cv2.imshow("Recording Facial & Body Language...", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ✅ Analyze Final Emotion Result
# cap.release()
# cv2.destroyAllWindows()

# # ✅ Determine Most Frequent Emotion & Body Posture
# final_emotion = Counter(emotion_predictions).most_common(1)[0][0]
# final_body_posture = Counter(body_postures).most_common(1)[0][0] if body_postures else "Neutral"

# print(f"\n✅ Final Detected Emotion: {final_emotion}")
# print(f"✅ Final Body Posture: {final_body_posture}")

# # ✅ 🎶 Music Selection Based on Mood
# music_genres = {
#     "Happy": ["Pop", "Dance", "Upbeat Jazz"],
#     "Sad": ["Soft Piano", "Acoustic", "Lo-Fi"],
#     "Angry": ["Rock", "Heavy Metal", "Hard Hip-Hop"],
#     "Surprise": ["Electronic", "Fast Jazz", "Funky"],
#     "Fear": ["Chill Lo-Fi", "Classical", "Ambient"],
#     "Disgust": ["Smooth Jazz", "R&B", "Soul"],
#     "Neutral": ["Indie", "Soft Rock", "Chill Pop"]
# }

# if final_body_posture in ["Slouched (Uncomfortable)", "Defensive / Closed-Off"]:
#     print("🟠 Adjusting music for stress relief...")
#     selected_genre = "Chill Lo-Fi"
# else:
#     selected_genre = random.choice(music_genres[final_emotion])

# print(f"🎵 Recommended Music Genre: {selected_genre}")

# import cv2
# import torch
# from feat.detector import Detector
# import threading
# import time

# # Load py-feat detector (Ensure CUDA is available for GPU acceleration)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# detector = Detector(device=device)

# # Start webcam
# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS

# frame_skip = 3  # Process every 3rd frame to increase FPS
# frame_count = 0

# def process_frame(frame):
#     """Process a single frame asynchronously."""
#     global frame_count

#     if frame is None:
#         return

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         return  # Skip frames to improve speed

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

#     faces = detector.detect_faces(frame_rgb)
#     if faces:
#         landmarks = detector.detect_landmarks(frame_rgb, faces)
#         print(f"Detected Landmarks: {landmarks}")
#         emotions = detector.detect_emotions(frame_rgb, faces, landmarks)

#     if emotions:
#         emotion_labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
#         emotion_dict = dict(zip(emotion_labels, emotions[0]))

#         # Only consider emotions with confidence > 0.3
#         filtered_emotions = {k: v.mean() for k, v in emotion_dict.items() if v.mean() > 0.3}

#         if filtered_emotions:
#             dominant_emotion = max(filtered_emotions, key=filtered_emotions.get)
#             print(f"Dominant Emotion: {dominant_emotion}")
#         else:
#             print("No strong emotion detected.")


# # Run continuously
# while cap.isOpened():
#     start_time = time.time()  # Track processing time

#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Use threading for async processing
#     threading.Thread(target=process_frame, args=(frame,)).start()

#     # Show webcam feed
#     cv2.imshow("Real-Time Emotion Detection", frame)

#     # Stop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     # Print FPS
#     fps = 1.0 / (time.time() - start_time)
#     print(f"FPS: {fps:.2f}")

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import time
# from collections import Counter
# import random

# # ✅ Load the Best Model
# MODEL_PATH = "../models/fine_tuned_Custom_CNN.keras"  # Change as needed
# model = tf.keras.models.load_model(MODEL_PATH)

# # ✅ Define Emotion Classes (Ensure it matches model output)
# EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]

# # ✅ Detect Model Input Type (Grayscale or RGB)
# input_shape = model.input_shape
# img_size = input_shape[1:3]  # Extracts (height, width)
# color_channels = input_shape[-1]  # Extracts 1 (Grayscale) or 3 (RGB)
# use_rgb = color_channels == 3  # ✅ True if model needs RGB

# # ✅ Initialize MediaPipe Face & Pose Detector
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# mp_pose = mp.solutions.pose
# pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # ✅ OpenCV Webcam Capture
# cap = cv2.VideoCapture(0)

# # ✅ User Customization: Set Recording Time
# RECORDING_TIME = 40  # ⏳ Set recording duration in seconds
# FRAME_RATE = 10  # 🎥 Number of frames per second
# frames_to_capture = RECORDING_TIME * FRAME_RATE
# emotion_predictions = []  # Stores detected emotions
# body_postures = []  # Stores detected body postures

# print(f"\n🎥 Recording Facial & Body Expressions for {RECORDING_TIME} Seconds...")

# start_time = time.time()

# while time.time() - start_time < RECORDING_TIME:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # ✅ Convert Frame to RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # ✅ Detect Faces
#     face_results = face_detection.process(rgb_frame)

#     # ✅ Detect Body Pose
#     pose_results = pose_detector.process(rgb_frame)

#     detected_emotion = "Neutral"  # Default if no face is detected
#     body_language = "Neutral"

#     if face_results.detections:
#         for detection in face_results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

#             # ✅ Extract Face
#             face = frame[y:y+h, x:x+w]

#             if face.shape[0] > 0 and face.shape[1] > 0:
#                 # ✅ Convert to Grayscale if Needed
#                 if use_rgb:
#                     face_resized = cv2.resize(face, img_size)
#                 else:
#                     face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                     face_resized = cv2.resize(face_gray, img_size)

#                 # ✅ Normalize and Reshape
#                 face_normalized = face_resized / 255.0
#                 face_reshaped = np.expand_dims(face_normalized, axis=0)
#                 if not use_rgb:
#                     face_reshaped = np.expand_dims(face_reshaped, axis=-1)

#                 # ✅ Predict Emotion with Error Handling
#                 prediction = model.predict(face_reshaped, verbose=0)

#                 if prediction.shape[1] == len(EMOTIONS):  # Ensure model outputs the correct number of classes
#                     emotion_idx = np.argmax(prediction)
#                     confidence = np.max(prediction)

#                     detected_emotion = EMOTIONS[emotion_idx]
#                     emotion_predictions.append(detected_emotion)

#                     # ✅ Display Emotion in Frame
#                     emotion_text = f"{detected_emotion} ({confidence*100:.1f}%)"
#                     cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 else:
#                     print(f"⚠️ Model output size mismatch: Expected {len(EMOTIONS)}, got {prediction.shape[1]}")

#     # ✅ Improved Body Language Detection
#     if pose_results.pose_landmarks:
#         landmarks = pose_results.pose_landmarks.landmark

#         left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
#         right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

#         # ✅ Posture Analysis
#         shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

#         if shoulder_diff > 0.05:
#             body_language = "Slouched (Uncomfortable)"
#         else:
#             body_language = "Upright (Neutral)"

#         # ✅ Hand Movements
#         hand_above_shoulder = (left_hand.y < left_shoulder.y) or (right_hand.y < right_shoulder.y)

#         if hand_above_shoulder:
#             body_language = "Excited / Surprised"
#         elif (left_hand.y > left_shoulder.y + 0.2) and (right_hand.y > right_shoulder.y + 0.2):
#             body_language = "Defensive / Closed-Off"
#         elif (left_hand.y > left_shoulder.y) and (right_hand.y > right_shoulder.y):
#             body_language = "Uncertain / Thinking"

#         body_postures.append(body_language)
#         cv2.putText(frame, body_language, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#     # ✅ Show Webcam Feed
#     cv2.imshow("Recording Facial & Body Language...", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ✅ Analyze Final Emotion Result
# cap.release()
# cv2.destroyAllWindows()

# # ✅ Determine Most Frequent Emotion & Body Posture
# final_emotion = Counter(emotion_predictions).most_common(1)[0][0]
# final_body_posture = Counter(body_postures).most_common(1)[0][0] if body_postures else "Neutral"

# print(f"\n✅ Final Detected Emotion: {final_emotion}")
# print(f"✅ Final Body Posture: {final_body_posture}")

# # ✅ 🎶 Music Selection Based on Mood
# music_genres = {
#     "Happy": ["Pop", "Dance", "Upbeat Jazz"],
#     "Sad": ["Soft Piano", "Acoustic", "Lo-Fi"],
#     "Angry": ["Rock", "Heavy Metal", "Hard Hip-Hop"],
#     "Surprise": ["Electronic", "Fast Jazz", "Funky"],
#     "Fear": ["Chill Lo-Fi", "Classical", "Ambient"],
#     "Disgust": ["Smooth Jazz", "R&B", "Soul"],
#     "Neutral": ["Indie", "Soft Rock", "Chill Pop"]
# }

# if final_body_posture in ["Slouched (Uncomfortable)", "Defensive / Closed-Off"]:
#     print("🟠 Adjusting music for stress relief...")
#     selected_genre = "Chill Lo-Fi"
# else:
#     selected_genre = random.choice(music_genres[final_emotion])

# print(f"🎵 Recommended Music Genre: {selected_genre}")
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import random
from collections import Counter
import tkinter as tk
from tkinter import ttk

# ✅ Define Available Models
MODEL_PATHS = {
    "CNN": "../models/fine_tuned_Custom_CNN.keras",
    "VGG19": "../models/fine_tuned_VGG19.keras",
    "ResNet50": "../models/fine_tuned_ResNet50.keras",
    "MobileNetV2": "../models/fine_tuned_MobileNetV2.keras"
}

# ✅ Default Model Selection
selected_model_name = "CNN"
model = tf.keras.models.load_model(MODEL_PATHS[selected_model_name])

# ✅ Emotion Classes
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]

# ✅ Detect Model Input Type
def detect_input_type():
    global img_size, use_rgb
    input_shape = model.input_shape
    img_size = input_shape[1:3]  # Extract (height, width)
    color_channels = input_shape[-1]  # Extract 1 (Grayscale) or 3 (RGB)
    use_rgb = color_channels == 3

detect_input_type()

# ✅ GUI for Model Selection
def switch_model(event):
    global model, selected_model_name
    selected_model_name = model_var.get()
    print(f"\n🔄 Switching to {selected_model_name} Model...\n")
    model = tf.keras.models.load_model(MODEL_PATHS[selected_model_name])
    detect_input_type()  # Update input type for new model

# ✅ Create Tkinter Window (Non-blocking)
window = tk.Tk()
window.title("Switch AI Model")
window.geometry("300x150")

label = tk.Label(window, text="Select Emotion Model:", font=("Arial", 12))
label.pack(pady=10)

model_var = tk.StringVar(value=selected_model_name)
dropdown = ttk.Combobox(window, textvariable=model_var, values=list(MODEL_PATHS.keys()), state="readonly")
dropdown.pack()
dropdown.bind("<<ComboboxSelected>>", switch_model)

# ✅ Start Emotion Detection on Button Click
def start_session():
    global model
    print(f"\n🎥 Starting 40-Second Emotion & Body Posture Detection with {selected_model_name}...\n")

    # ✅ Initialize MediaPipe Face & Pose Detector
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ✅ OpenCV Webcam Capture
    cap = cv2.VideoCapture(0)

    # ✅ Set 40-Second Timer
    RECORDING_TIME = 40
    start_time = time.time()
    emotion_predictions = []
    body_postures = []

    while time.time() - start_time < RECORDING_TIME:
        ret, frame = cap.read()
        if not ret:
            break

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

                face = frame[y:y+h, x:x+w]

                if face.shape[0] > 0 and face.shape[1] > 0:
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
                    if prediction.shape[1] == len(EMOTIONS):
                        emotion_idx = np.argmax(prediction)
                        confidence = np.max(prediction)

                        detected_emotion = EMOTIONS[emotion_idx]
                        emotion_predictions.append(detected_emotion)

                        # ✅ Display Emotion
                        emotion_text = f"{detected_emotion} ({confidence*100:.1f}%)"
                        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        print(f"⚠️ Model output mismatch: Expected {len(EMOTIONS)}, got {prediction.shape[1]}")

        # ✅ Show Webcam Feed
        cv2.imshow("Live Emotion & Body Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ✅ Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # ✅ Determine Most Frequent Emotion & Body Posture
    final_emotion = Counter(emotion_predictions).most_common(1)[0][0] if emotion_predictions else "Neutral"
    final_body_posture = Counter(body_postures).most_common(1)[0][0] if body_postures else "Neutral"

    print(f"\n✅ Final Detected Emotion: {final_emotion}")
    print(f"✅ Final Body Posture: {final_body_posture}")

    # ✅ 🎶 Suggest Music Based on Mood
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
        selected_genre = random.choice(music_genres[final_emotion])

    print(f"🎵 Recommended Music Genre: {selected_genre}")

# ✅ Button to Start 40-Second Detection
start_button = tk.Button(window, text="Start 40-Second Session", command=start_session, font=("Arial", 12), bg="lightgreen")
start_button.pack(pady=10)

# ✅ Keep GUI Open
window.mainloop()
