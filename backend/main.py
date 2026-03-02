"""
rPPG Health Monitor - FastAPI Backend Server
Real-time facial recognition and vital sign extraction via WebSocket.
"""

import asyncio
import base64
import json
import time
import logging
from typing import Optional, Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rppg_processor import RPPGProcessor
from motion_analyzer import MotionAnalyzer
from luminance_adjuster import LuminanceAdjuster
from face_recognition_engine import FaceRecognitionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="rPPG Health Monitor API",
    description="Real-time facial recognition and vital sign extraction",
    version="1.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionState:
    """Manages per-session processing state."""

    def __init__(self, fps: float = 30.0):
        self.rppg = RPPGProcessor(fps=fps, buffer_seconds=12.0)
        self.motion = MotionAnalyzer(buffer_size=60)
        self.luminance = LuminanceAdjuster()
        self.face_engine = FaceRecognitionEngine()
        self.face_engine.load_known_faces()
        self.is_active = False
        self.start_time: Optional[float] = None
        self.max_duration = 60.0  # 1 minute max scan

    def reset(self):
        self.rppg.reset()
        self.motion.reset()
        self.luminance.reset()
        self.is_active = False
        self.start_time = None

    def should_stop(self) -> bool:
        if self.start_time and self.max_duration:
            return (time.time() - self.start_time) >= self.max_duration
        return False


class RegisterFaceRequest(BaseModel):
    name: str
    image_data: str  # base64 encoded image


# REST Endpoints
@app.get("/")
async def root():
    return {"status": "ok", "service": "rPPG Health Monitor", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/api/register-face")
async def register_face(request: RegisterFaceRequest):
    """Register a new face for recognition."""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(request.image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        engine = FaceRecognitionEngine()
        face_data = engine.detect_face(frame)

        if not face_data or not face_data.get("detected"):
            raise HTTPException(status_code=400, detail="No face detected in image")

        success = engine.register_face(request.name, face_data["embedding"])
        engine.cleanup()

        if success:
            return {"status": "success", "name": request.name}
        else:
            raise HTTPException(status_code=500, detail="Failed to register face")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/known-faces")
async def get_known_faces():
    """List all registered faces."""
    engine = FaceRecognitionEngine()
    engine.load_known_faces()
    names = engine.known_face_names
    engine.cleanup()
    return {"faces": names, "count": len(names)}


# WebSocket Endpoint - Real-time Processing
@app.websocket("/ws/scan")
async def websocket_scan(websocket: WebSocket):
    await websocket.accept()
    session = SessionState(fps=10.0) # Assume 10fps from client

    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type", "frame")

            if msg_type == "start":
                session.reset()
                session.is_active = True
                session.start_time = time.time()
                await websocket.send_json({
                    "type": "status",
                    "status": "started",
                    "message": "Scan started. Hold still and face the camera.",
                })
                continue

            elif msg_type == "stop":
                vitals = session.rppg.get_vitals()
                session.is_active = False
                await websocket.send_json({
                    "type": "results",
                    "status": "completed",
                    "vitals": vitals,
                    "motion_stats": session.motion.get_stats(),
                    "signal_quality": session.rppg.get_signal_quality(),
                })
                session.reset()
                continue

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                continue

            elif msg_type == "frame":
                if not session.is_active:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Scan not started. Send 'start' message first.",
                    })
                    continue

                # 1. Decode frame
                frame_data = message.get("data", "")
                if not frame_data:
                    continue

                try:
                    if "," in frame_data:
                        frame_data = frame_data.split(",")[1]

                    img_bytes = base64.b64decode(frame_data)
                    frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is None:
                        continue
                except Exception:
                    continue

                # 2. Process frame
                timestamp = time.time()
                elapsed = round(timestamp - session.start_time, 1) if session.start_time else 0
                
                # Check for 60s limit
                if elapsed >= 60.0:
                    vitals = session.rppg.get_vitals()
                    await websocket.send_json({
                        "type": "results",
                        "status": "time_limit_reached",
                        "vitals": vitals,
                        "elapsed_time": 60.0,
                    })
                    session.is_active = False
                    session.reset()
                    continue

                # Luminance adjustment
                adjusted_frame = session.luminance.full_adjustment(frame)

                # Face detection
                face_data = session.face_engine.detect_face(frame)
                
                vitals = None
                motion_data = None
                identity = None

                if face_data and face_data.get("detected"):
                    # Motion analysis
                    landmarks = face_data.get("landmarks")
                    motion_data = session.motion.analyze_frame(adjusted_frame, landmarks)

                    # rPPG signal processing
                    if not session.motion.should_skip_frame():
                        rppg_roi = face_data.get("rppg_roi")
                        if rppg_roi is not None and rppg_roi.size > 0:
                            session.rppg.add_frame_roi(rppg_roi, timestamp)

                    # Face identification
                    embedding = face_data.get("embedding")
                    if embedding is not None:
                        identity = session.face_engine.identify_face(embedding)

                    # Get current vitals
                    vitals = session.rppg.get_vitals()

                # ALWAYS build and send a response so the timer keeps moving
                response = {
                    "type": "update",
                    "timestamp": timestamp,
                    "elapsed_time": elapsed,
                    "face": {
                        "detected": face_data.get("detected", False) if face_data else False,
                        "bbox": face_data.get("bbox") if face_data else None,
                        "confidence": face_data.get("confidence", 0) if face_data else 0,
                        "identity": identity,
                    },
                    "motion": {
                        "score": motion_data["motion_score"] if motion_data else 0,
                        "level": motion_data["motion_level"] if motion_data else "unknown",
                        "is_stable": motion_data["is_stable"] if motion_data else True,
                    } if motion_data else None,
                    "vitals": vitals if vitals else session.rppg.get_vitals(),
                    "signal_quality": session.rppg.get_signal_quality(),
                }

                await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        session.face_engine.cleanup()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        session.face_engine.cleanup()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
