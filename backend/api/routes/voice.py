# backend/api/routes/voice.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.services.voice_command_service import handle_voice_command
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/voice-command")
async def voice_command(image: UploadFile = File(...), transcript: str = Form(...)):
    """
    Accepts a snapshot + spoken transcript, classifies intent, routes to the
    matching feature, and returns a spoken-ready response.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        image_bytes = await image.read()
        result = handle_voice_command(image_bytes, transcript)
    except Exception as e:
        logger.error(f"Voice command failed: {e}")
        raise HTTPException(status_code=500, detail="Voice command processing failed.")

    return result