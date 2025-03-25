import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from utils.model_loader import load_model
from collections import Counter
import time
from pydantic import BaseModel


# ✅ Load default model
model = load_model("CNN")

# ✅ Define emotions
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]

# ✅ FastAPI Router
router = APIRouter()

@router.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...), model_name: str = "CNN"):
    """Processes an image and predicts the facial emotion."""
    global model

    # ✅ Load model dynamically
    model = load_model(model_name)

    try:
        # ✅ Read the uploaded file
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ✅ Convert to grayscale if needed
        img_size = model.input_shape[1:3]
        color_channels = model.input_shape[-1]

        if color_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_resized = cv2.resize(image, img_size) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        if color_channels == 1:
            image_resized = np.expand_dims(image_resized, axis=-1)

        # ✅ Predict emotion
        prediction = model.predict(image_resized, verbose=0)
        emotion_idx = np.argmax(prediction)
        detected_emotion = EMOTIONS[emotion_idx]
        confidence = np.max(prediction)

        return {"emotion": detected_emotion, "confidence": f"{confidence:.2%}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error processing image: {str(e)}")



# ✅ Pydantic Schema for API Documentation
class EmotionResponse(BaseModel):
    detected_emotion: str
    frames_analyzed: int

@router.get("/record-expression/", response_model=EmotionResponse, summary="Detect Emotion from Webcam")
async def record_expression(duration: int = 5, model_name: str = "CNN"):
    """
    **Records facial expressions from the webcam for `duration` seconds**  
    and returns the most frequently detected emotion.
    
    - **duration**: Recording time in seconds (default: 5)
    - **model_name**: AI Model used for emotion detection (default: "CNN")
    
    **Returns**:
    - `detected_emotion`: The most frequent emotion detected
    - `frames_analyzed`: Number of frames processed
    """
    model = load_model(model_name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "❌ Webcam not found!"}

    start_time = time.time()
    emotion_predictions = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        img_size = model.input_shape[1:3]
        color_channels = model.input_shape[-1]

        if color_channels == 1:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(frame_gray, img_size)
        else:
            face_resized = cv2.resize(frame, img_size)

        face_normalized = face_resized / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=0)

        if color_channels == 1:
            face_reshaped = np.expand_dims(face_reshaped, axis=-1)

        # Predict Emotion
        prediction = model.predict(face_reshaped, verbose=0)
        emotion_idx = np.argmax(prediction)
        detected_emotion = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"][emotion_idx]
        emotion_predictions.append(detected_emotion)

        time.sleep(1)

    cap.release()
    final_emotion = Counter(emotion_predictions).most_common(1)[0][0] if emotion_predictions else "Neutral"
    
    return {"detected_emotion": final_emotion, "frames_analyzed": len(emotion_predictions)}

