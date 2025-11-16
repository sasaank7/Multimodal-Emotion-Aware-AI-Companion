"""
FastAPI Backend for Emotion-Aware AI Companion.
Provides REST API and WebSocket endpoints for scalable deployment.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import uuid
from datetime import datetime
import io
import base64

from ..emotion_detection.text_sentiment import TextSentimentAnalyzer
from ..emotion_detection.multimodal_fusion import MultimodalEmotionFusion
from ..llm_engine.adaptive_response import AdaptiveResponseSystem
from ..utils.personalization import UserProfile
from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion-Aware AI Companion API",
    description="REST API and WebSocket interface for multimodal emotion-aware AI",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global instances (in production, use dependency injection and pooling)
_sentiment_analyzer = None
_fusion_engine = None
_adaptive_system = None
_active_sessions: Dict[str, Dict] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class EmotionDetectionRequest(BaseModel):
    """Request model for emotion detection."""
    text: Optional[str] = None
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    user_id: Optional[str] = "default"


class ChatRequest(BaseModel):
    """Request model for chat interaction."""
    message: str
    user_id: Optional[str] = "default"
    emotion_context: Optional[Dict] = None
    include_voice: bool = False


class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str
    emotion_context: Optional[Dict] = None
    response_mode: str
    confidence: float
    audio_base64: Optional[str] = None
    timestamp: str


class EmotionHistoryRequest(BaseModel):
    """Request model for emotion history."""
    user_id: str = "default"
    days: int = 7
    limit: Optional[int] = None


class UserStatsResponse(BaseModel):
    """Response model for user statistics."""
    user_id: str
    total_interactions: int
    mood_patterns: Dict
    interaction_stats: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global _sentiment_analyzer, _fusion_engine, _adaptive_system

    logger.info("Starting Emotion-Aware AI Companion API...")

    try:
        _sentiment_analyzer = TextSentimentAnalyzer()
        _fusion_engine = MultimodalEmotionFusion()
        _adaptive_system = AdaptiveResponseSystem()

        logger.info("API services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _adaptive_system

    logger.info("Shutting down API...")

    if _adaptive_system:
        _adaptive_system.cleanup()

    logger.info("API shutdown complete")


# ============================================================================
# Helper Functions
# ============================================================================

def get_user_profile(user_id: str) -> UserProfile:
    """Get or create user profile."""
    return UserProfile(user_id=user_id)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token (placeholder)."""
    # In production, implement proper JWT verification
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return credentials.credentials


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        services={
            "sentiment_analyzer": "running" if _sentiment_analyzer else "stopped",
            "fusion_engine": "running" if _fusion_engine else "stopped",
            "adaptive_system": "running" if _adaptive_system else "stopped"
        }
    )


@app.get("/api/v1/status")
async def get_status():
    """Get API status and metrics."""
    return {
        "status": "running",
        "active_sessions": len(_active_sessions),
        "uptime": "calculated_uptime_here",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Emotion Detection Endpoints
# ============================================================================

@app.post("/api/v1/emotion/detect")
async def detect_emotion(request: EmotionDetectionRequest):
    """
    Detect emotions from multimodal input.
    """
    try:
        text_emotion = None
        face_emotion = None
        voice_emotion = None

        # Process text
        if request.text:
            text_emotion = _sentiment_analyzer.analyze_sentiment(request.text)

        # Process image (facial emotion)
        if request.image_base64:
            # Decode base64 image and process
            # Implementation here...
            pass

        # Process audio (voice emotion)
        if request.audio_base64:
            # Decode base64 audio and process
            # Implementation here...
            pass

        # Fuse emotions
        fused_emotion = _fusion_engine.fuse_emotions(
            face_emotion=face_emotion,
            voice_emotion=voice_emotion,
            text_emotion=text_emotion
        )

        return {
            "fused_emotion": fused_emotion,
            "individual_emotions": {
                "text": text_emotion,
                "face": face_emotion,
                "voice": voice_emotion
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/emotion/text")
async def analyze_text_sentiment(text: str, user_id: str = "default"):
    """Analyze text sentiment."""
    try:
        result = _sentiment_analyzer.analyze_sentiment(text)

        # Track in user profile
        profile = get_user_profile(user_id)
        if result:
            profile.add_mood_entry(result)
        profile.close()

        return result or {"error": "No sentiment detected"}

    except Exception as e:
        logger.error(f"Text sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message and generate adaptive response.
    """
    try:
        # Analyze text emotion
        text_emotion = _sentiment_analyzer.analyze_sentiment(request.message)

        # Generate response
        result = _adaptive_system.generate_response(
            user_message=request.message,
            face_emotion=None,
            voice_emotion=None,
            text_emotion=text_emotion
        )

        # Track in user profile
        profile = get_user_profile(request.user_id)
        profile.add_interaction(
            user_message=request.message,
            assistant_response=result['response'],
            emotion_context=result['emotion_context'],
            response_mode=result['response_mode']
        )
        profile.close()

        # Generate voice if requested
        audio_base64 = None
        if request.include_voice:
            # Generate TTS and encode
            # Implementation here...
            pass

        return ChatResponse(
            response=result['response'],
            emotion_context=result['emotion_context'],
            response_mode=result['response_mode'],
            confidence=result['confidence'],
            audio_base64=audio_base64,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response (for real-time updates).
    """
    async def generate():
        try:
            # This would implement streaming response from LLM
            # For now, return full response
            result = await chat(request)
            yield f"data: {result.json()}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {{'error': '{str(e)}'}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ============================================================================
# User Profile Endpoints
# ============================================================================

@app.get("/api/v1/user/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(user_id: str, days: int = 7):
    """Get user statistics and analytics."""
    try:
        profile = get_user_profile(user_id)

        context = profile.get_personalization_context()
        mood_patterns = profile.get_mood_patterns(days=days)
        interaction_stats = profile.get_interaction_stats(days=days)

        profile.close()

        return UserStatsResponse(
            user_id=user_id,
            total_interactions=context.get('total_interactions', 0),
            mood_patterns=mood_patterns,
            interaction_stats=interaction_stats
        )

    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/user/{user_id}/history")
async def get_emotion_history(user_id: str, days: int = 7, limit: Optional[int] = None):
    """Get user emotion history."""
    try:
        profile = get_user_profile(user_id)
        history = profile.get_mood_history(days=days, limit=limit)
        profile.close()

        return {
            "user_id": user_id,
            "history": history,
            "count": len(history)
        }

    except Exception as e:
        logger.error(f"Failed to get emotion history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/user/{user_id}/preference")
async def set_user_preference(user_id: str, key: str, value: Any):
    """Set user preference."""
    try:
        profile = get_user_profile(user_id)
        profile.set_preference(key, value)
        profile.close()

        return {"status": "success", "key": key, "value": value}

    except Exception as e:
        logger.error(f"Failed to set preference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/user/{user_id}/export")
async def export_user_data(user_id: str):
    """Export user data."""
    try:
        profile = get_user_profile(user_id)
        filepath = profile.export_data()
        profile.close()

        # In production, return file download
        return {
            "status": "success",
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and store connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        """Remove connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific connection."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

    async def broadcast(self, message: dict):
        """Broadcast to all connections."""
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time communication.
    """
    await manager.connect(websocket, session_id)

    # Initialize session
    _active_sessions[session_id] = {
        "created_at": datetime.now(),
        "user_id": "default",
        "message_count": 0
    }

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            message_type = data.get("type", "chat")

            if message_type == "chat":
                # Process chat message
                request = ChatRequest(**data.get("data", {}))

                # Generate response
                text_emotion = _sentiment_analyzer.analyze_sentiment(request.message)

                result = _adaptive_system.generate_response(
                    user_message=request.message,
                    text_emotion=text_emotion
                )

                # Send response
                await manager.send_personal_message(
                    {
                        "type": "chat_response",
                        "data": {
                            "response": result['response'],
                            "emotion_context": result['emotion_context'],
                            "response_mode": result['response_mode'],
                            "timestamp": datetime.now().isoformat()
                        }
                    },
                    session_id
                )

                # Update session
                _active_sessions[session_id]["message_count"] += 1

            elif message_type == "emotion_update":
                # Handle emotion detection update
                emotion_data = data.get("data", {})

                await manager.send_personal_message(
                    {
                        "type": "emotion_acknowledged",
                        "data": emotion_data
                    },
                    session_id
                )

            elif message_type == "ping":
                # Heartbeat
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now().isoformat()},
                    session_id
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        if session_id in _active_sessions:
            del _active_sessions[session_id]

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)


# ============================================================================
# Analytics Endpoints
# ============================================================================

@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(days: int = 7):
    """Get system-wide analytics summary."""
    return {
        "total_sessions": len(_active_sessions),
        "active_connections": len(manager.active_connections),
        "period_days": days,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Configuration Endpoints
# ============================================================================

@app.get("/api/v1/config")
async def get_config_info():
    """Get public configuration information."""
    config = get_config()

    return {
        "emotion_detection": {
            "modalities": ["face", "voice", "text"],
            "fusion_strategy": config.get("fusion.strategy")
        },
        "response_modes": list(config.get("llm.modes", {}).keys()),
        "features": config.get("features", {})
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
