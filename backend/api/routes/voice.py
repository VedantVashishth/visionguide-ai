# backend/api/routes/voice.py
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.services.voice_command_service import handle_voice_command
from backend.services.face_service import save_person_faces, list_known_faces, forget_face
from backend.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/voice-command")
async def voice_command(image: UploadFile = File(...), transcript: str = Form(...)):
    """
    Accepts a snapshot + spoken transcript, classifies intent, routes to the
    matching feature, and returns a spoken-ready response.

    If the response has "awaiting_name": true, the frontend should listen for
    the next thing the user says and send it to /save-face instead of
    routing it back through this endpoint.
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


@router.post("/save-face")
async def save_face(
    images: List[UploadFile] = File(...),
    name: str = Form(...),
):
    """
    Follow-up step after 'remember_person' intent — saves the currently
    visible face under the given name, using OpenCV's local LBPH recognizer.
    Accepts multiple snapshots for better accuracy. No cloud calls.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    for upload in images:
        if not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded files must be images.")

    try:
        image_bytes_list = [await upload.read() for upload in images]
        result = save_person_faces(image_bytes_list, name)
    except ValueError as e:
        logger.warning(f"Bad image upload for save-face: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Save face failed: {e}")
        raise HTTPException(status_code=500, detail="Saving face failed.")

    return result


@router.get("/known-faces")
async def known_faces():
    """Return names currently stored in the local face database."""
    try:
        return list_known_faces()
    except Exception as e:
        logger.error(f"List known faces failed: {e}")
        raise HTTPException(status_code=500, detail="Could not list known faces.")


@router.delete("/known-faces/{name}")
async def delete_known_face(name: str):
    """Remove a remembered person from the local face database."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    try:
        return forget_face(name)
    except Exception as e:
        logger.error(f"Delete known face failed: {e}")
        raise HTTPException(status_code=500, detail="Could not delete known face.")
