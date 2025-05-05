# backend/routes/analytics.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
from datetime import datetime
import json
import os
import requests

router = APIRouter()

GOOGLE_SHEET_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzz74ZGYjDh1HBWPYqLFZGdaBSmCn8eUJKLaLXR29WUO84sQnH0ViVcwq1abQfnaML9uA/exec"

class EmotionLog(BaseModel):
    emotion: str
    confidence: float
    room: str
    userId: str = "anonymous"

@router.post("/log-emotion")
def log_emotion(data: EmotionLog):
    try:
        response = requests.post(GOOGLE_SHEET_WEBHOOK_URL, json=data.dict())
        return {"status": "sent", "sheet_response": response.text}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
# LOG_FILE = "emotion_logs.json"

# class EmotionLog(BaseModel):
#     emotion: str
#     confidence: float
#     room: str
#     timestamp: datetime

# @router.post("/log-emotion")
# async def log_emotion(data: EmotionLog):
#     log_entry = data.dict()
#     log_entry["timestamp"] = data.timestamp.isoformat()

#     if os.path.exists(LOG_FILE):
#         with open(LOG_FILE, "r") as f:
#             logs = json.load(f)
#     else:
#         logs = []

#     logs.append(log_entry)

#     with open(LOG_FILE, "w") as f:
#         json.dump(logs, f, indent=2)

#     return {"status": "success", "message": "Emotion logged"}
