from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from services.emotion_service import router as emotion_router
from services.music_services import router as music_router

app = FastAPI(title="Online Museum AI Backend", version="1.0")

# Allow requests from the frontend 
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://musemind-navy.vercel.app/"

]


# ✅ CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include API Routes
app.include_router(emotion_router, prefix="/api")
app.include_router(music_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "🎭 Welcome to the Online Museum AI Backend!"}

@app.on_event("shutdown")
def shutdown_event():
    print("🛑 FastAPI server is shutting down properly...")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)