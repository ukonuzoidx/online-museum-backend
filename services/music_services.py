from fastapi import APIRouter
import random

# ✅ Music Mapping
MUSIC_GENRES = {
    "Happy": ["Pop", "Dance", "Upbeat Jazz"],
    "Sad": ["Soft Piano", "Acoustic", "Lo-Fi"],
    "Angry": ["Rock", "Heavy Metal", "Hard Hip-Hop"],
    "Surprise": ["Electronic", "Fast Jazz", "Funky"],
    "Fear": ["Chill Lo-Fi", "Classical", "Ambient"],
    "Disgust": ["Smooth Jazz", "R&B", "Soul"],
    "Neutral": ["Indie", "Soft Rock", "Chill Pop"]
}

# ✅ FastAPI Router
router = APIRouter()

@router.get("/recommend-music/")
async def recommend_music(emotion: str):
    """Returns a music recommendation based on emotion."""
    if emotion not in MUSIC_GENRES:
        return {"error": "Emotion not recognized."}

    selected_genre = random.choice(MUSIC_GENRES[emotion])
    return {"emotion": emotion, "recommended_music": selected_genre}
