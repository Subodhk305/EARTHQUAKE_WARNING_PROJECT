"""WebSocket manager for real-time seismic alerts."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
ws_router = APIRouter()


class AlertManager:
    """Manages active WebSocket connections and broadcasts alerts."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info("✅ WebSocket connected. Total: %d", len(self.active))
        
        # Send welcome message
        try:
            await ws.send_json({
                "type": "connection",
                "message": "Connected to earthquake alert service",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info("❌ WebSocket disconnected. Total: %d", len(self.active))

    async def broadcast(self, payload: dict):
        if not self.active:
            return
        message = json.dumps(payload)
        dead = set()
        for ws in self.active.copy():
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to client: {e}")
                dead.add(ws)
        for ws in dead:
            self.disconnect(ws)


alert_manager = AlertManager()


@ws_router.websocket("/ws")
async def websocket_alerts(websocket: WebSocket):
    """
    Real-time alert endpoint.
    Clients connect here to receive high-risk earthquake alerts.
    """
    await alert_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; server pushes data
            # Receive any client messages if needed
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")
    except WebSocketDisconnect:
        alert_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        alert_manager.disconnect(websocket)