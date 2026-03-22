# earthquake_service/api/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import json
from typing import Dict, Set
from datetime import datetime

# Import the router correctly
from earthquake_service.api.routes import router
from earthquake_service.services.model_loader import ModelLoader
from earthquake_service.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = "default"):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        logger.info(f"✅ WebSocket connected: {client_id}")
        return client_id
    
    def disconnect(self, websocket: WebSocket, client_id: str = "default"):
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        logger.info(f"❌ WebSocket disconnected: {client_id}")
    
    async def send_message(self, message: str, client_id: str = "default"):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to {client_id}: {e}")
    
    async def broadcast(self, message: str):
        for client_id in list(self.active_connections.keys()):
            await self.send_message(message, client_id)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Earthquake Detection Service...")
    logger.info("=" * 60)
    
    # Load models on startup
    try:
        logger.info("📦 Loading models...")
        await ModelLoader.load_all()
        
        if ModelLoader.is_ready():
            logger.info("🎉 All models are ready for inference!")
        else:
            logger.warning("⚠️ Some models failed to load. Service may have limited functionality.")
            
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        logger.warning("⚠️ Service will run with limited functionality (predictions may not work)")
    
    yield  # Server runs here
    
    # Cleanup on shutdown
    logger.info("=" * 60)
    logger.info("🛑 Shutting down Earthquake Detection Service...")
    logger.info("=" * 60)

# Create FastAPI app with lifespan
app = FastAPI(
    title="Earthquake Detection API",
    version=settings.MODEL_VERSION,
    description="Real-time earthquake prediction using CNN-LSTM + XGBoost",
    lifespan=lifespan
)

# Configure CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - THIS IS THE KEY FIX
# The router already has prefix="/api/v1" in routes.py, so we include it directly
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Earthquake Detection API",
        "version": settings.MODEL_VERSION,
        "status": "running",
        "models_loaded": ModelLoader.is_ready(),
        "model_info": ModelLoader.get_model_info(),
        "websocket": "ws://localhost:8000/ws",
        "endpoints": {
            "predict": "/api/v1/predict",
            "historical": "/api/v1/historical",
            "model_metrics": "/api/v1/model-metrics",
            "health": "/api/v1/health"
        }
    }

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "models_ready": ModelLoader.is_ready(),
        "model_info": ModelLoader.get_model_info(),
        "timestamp": datetime.now().isoformat()
    }

# earthquake_service/api/main.py (update the websocket_endpoint function)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "default"):
    """WebSocket endpoint for real-time predictions"""
    await manager.connect(websocket, client_id)
    
    try:
        # Send connection confirmation
        await manager.send_message(json.dumps({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "message": "Connected to earthquake detection service",
            "models_ready": ModelLoader.is_ready(),
            "model_info": ModelLoader.get_model_info(),
            "timestamp": datetime.now().isoformat()
        }), client_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"📨 Received from {client_id}: {data[:200]}...")
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "")
                
                if message_type == "prediction" or message_type == "predict":
                    # Handle prediction request
                    if not ModelLoader.is_ready():
                        await manager.send_message(json.dumps({
                            "type": "error",
                            "error": "Models not loaded yet. Please try again later."
                        }), client_id)
                        continue
                    
                    # Make prediction
                    from earthquake_service.services.predictor import run_prediction
                    
                    result = await run_prediction(
                        latitude=message.get("latitude"),
                        longitude=message.get("longitude"),
                        location_name=message.get("location_name", ""),
                        radius_km=message.get("radius_km", 200),
                        include_waveform=message.get("include_waveform", False)
                    )
                    
                    # Convert to dict if it's a Pydantic model
                    if hasattr(result, 'model_dump'):
                        result_dict = result.model_dump()
                    else:
                        result_dict = result
                    
                    # Send prediction result
                    await manager.send_message(json.dumps({
                        "type": "prediction_result",
                        "data": result_dict,
                        "timestamp": datetime.now().isoformat()
                    }), client_id)
                    
                elif message_type == "ping":
                    # Respond to ping
                    await manager.send_message(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }), client_id)
                    
                elif message_type == "model_status":
                    # Send model status
                    await manager.send_message(json.dumps({
                        "type": "model_status",
                        "models_ready": ModelLoader.is_ready(),
                        "model_info": ModelLoader.get_model_info()
                    }), client_id)
                    
                elif message_type == "subscribe":
                    # Handle subscription requests
                    await manager.send_message(json.dumps({
                        "type": "subscribed",
                        "status": "success",
                        "message": f"Subscribed to updates for {message.get('location', 'all')}"
                    }), client_id)
                    
                else:
                    # Unknown message type - don't send error, just log
                    logger.warning(f"Unknown message type: {message_type}")
                    # Optionally send acknowledgment
                    await manager.send_message(json.dumps({
                        "type": "ack",
                        "status": "received",
                        "message_type": message_type
                    }), client_id)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                await manager.send_message(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON format"
                }), client_id)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await manager.send_message(json.dumps({
                    "type": "error",
                    "error": str(e)
                }), client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, client_id)