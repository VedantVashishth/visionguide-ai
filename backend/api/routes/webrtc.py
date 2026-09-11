"""In-memory WebRTC signaling relay (video never passes through the server)."""
import json
from collections import defaultdict
from typing import DefaultDict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
rooms: DefaultDict[str, dict[str, WebSocket]] = defaultdict(dict)
# A volunteer commonly opens the shared URL after the caller has created an
# offer. Keep those small signaling messages until that role connects.
pending_messages: DefaultDict[str, DefaultDict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
VALID_ROLES = {"sender", "receiver"}


@router.websocket("/ws/signaling/{room_id}/{role}")
async def signaling_endpoint(websocket: WebSocket, room_id: str, role: str):
    """Relay offers, answers, and ICE candidates for a single call room."""
    if role not in VALID_ROLES or not room_id or len(room_id) > 128:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    previous = rooms[room_id].get(role)
    rooms[room_id][role] = websocket
    if previous and previous is not websocket:
        await previous.close(code=1012)
    peer_role = "receiver" if role == "sender" else "sender"
    queued_messages = pending_messages[room_id].pop(role, [])
    for queued_message in queued_messages:
        await websocket.send_text(queued_message)
    logger.info("WebRTC %s joined room %s", role, room_id)
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if isinstance(message, dict):
                serialized = json.dumps(message)
                peer = rooms[room_id].get(peer_role)
                if peer:
                    await peer.send_text(serialized)
                else:
                    pending_messages[room_id][peer_role].append(serialized)
    except (WebSocketDisconnect, ValueError):
        pass
    finally:
        if rooms.get(room_id, {}).get(role) is websocket:
            del rooms[room_id][role]
            if not rooms[room_id]:
                del rooms[room_id]
        if not rooms.get(room_id) and not pending_messages.get(room_id):
            pending_messages.pop(room_id, None)
        logger.info("WebRTC %s left room %s", role, room_id)
