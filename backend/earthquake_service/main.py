"""
AI-Powered Earthquake Prediction Module
Main Application Entry Point
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse

from earthquake_service.api.routes import router as api_router
from earthquake_service.websocket.manager import ws_router
from earthquake_service.utils.database import init_db, close_db
from earthquake_service.utils.cache import init_redis, close_redis
from earthquake_service.utils.logger import setup_logging
from earthquake_service.services.model_loader import ModelLoader
from earthquake_service.config import settings

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("🚀 Starting Earthquake Prediction Service...")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Initialize Redis only if enabled
    if settings.USE_REDIS:
        try:
            await init_redis()
            logger.info("✅ Redis initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization failed: {e}")
    else:
        logger.info("ℹ️ Redis is disabled (USE_REDIS=false)")
    
    # Load ML models
    await ModelLoader.load_all()
    logger.info("✅ All models and services initialized.")
    
    # Log all routes after startup
    logger.info("=" * 50)
    logger.info("REGISTERED ROUTES:")
    for route in app.routes:
        methods = ",".join(route.methods) if hasattr(route, 'methods') else 'N/A'
        logger.info(f"  {route.path} [{methods}]")
    logger.info("=" * 50)
    
    yield
    
    logger.info("🛑 Shutting down Earthquake Prediction Service...")
    
    # Close database connections
    await close_db()
    
    # Close Redis only if enabled
    if settings.USE_REDIS:
        await close_redis()


app = FastAPI(
    title="Earthquake Prediction API",
    description=(
        "AI-Powered Earthquake Forecasting using CNN-LSTM-XGBoost hybrid architecture "
        "with IRIS FDSN waveform data and USGS historical catalog."
    ),
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/info", include_in_schema=True)
async def info():
    """Get API information and available endpoints."""
    return {
        "name": "Earthquake Prediction API",
        "version": settings.MODEL_VERSION,
        "model_name": settings.MODEL_NAME,
        "description": "AI-Powered Earthquake Forecasting using CNN-LSTM-XGBoost",
        "status": "operational",
        "redis_enabled": settings.USE_REDIS,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "info": "/info",
            "api": "/api/v1",
            "websocket": "/ws"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

# ── Routers ───────────────────────────────────────────────────────────────────
logger.info("📡 Mounting API router at /api/v1...")
app.include_router(api_router, prefix="/api/v1")
logger.info("🔌 Mounting WebSocket router...")
app.include_router(ws_router)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy", 
        "service": "earthquake-prediction",
        "version": settings.MODEL_VERSION,
        "models_loaded": ModelLoader.is_ready(),
        "redis_enabled": settings.USE_REDIS,
    }


@app.get("/routes", include_in_schema=False)
async def list_routes():
    """List all registered routes (for debugging)."""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods) if hasattr(route, "methods") else None
        })
    return {"routes": routes, "count": len(routes)}


@app.get("/landing", include_in_schema=False)
async def landing():
    """Simple HTML landing page."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Earthquake Prediction API</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                line-height: 1.6;
                color: #333;
                background: #0B0F19;
                color: #e2e8f0;
            }}
            h1 {{ color: #00D4FF; }}
            h2 {{ color: #94a3b8; margin-top: 30px; }}
            .endpoint {{
                background: #1A2540;
                padding: 10px 15px;
                border-radius: 5px;
                margin: 10px 0;
                border-left: 4px solid #00D4FF;
            }}
            .endpoint code {{
                background: #0f172a;
                padding: 3px 6px;
                border-radius: 3px;
                font-size: 1.1em;
                color: #00D4FF;
            }}
            .badge {{
                background: #00D4FF;
                color: #0B0F19;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 0.8em;
                margin-left: 10px;
                font-weight: bold;
            }}
            a {{ color: #00D4FF; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            hr {{ border-color: #1A2540; }}
        </style>
    </head>
    <body>
        <h1>🌍 Earthquake Prediction API</h1>
        <p>AI-Powered Earthquake Forecasting using CNN-LSTM-XGBoost hybrid architecture with IRIS FDSN waveform data and USGS historical catalog.</p>
        
        <div class="badge">v{settings.MODEL_VERSION}</div>
        
        <h2>📚 Documentation</h2>
        <div class="endpoint">
            <code>📘 /docs</code> - Interactive Swagger UI documentation
        </div>
        <div class="endpoint">
            <code>📕 /redoc</code> - ReDoc documentation
        </div>
        
        <h2>🔗 API Endpoints</h2>
        <div class="endpoint">
            <code>🔍 /health</code> - Health check
        </div>
        <div class="endpoint">
            <code>ℹ️ /info</code> - API information
        </div>
        <div class="endpoint">
            <code>📡 /api/v1</code> - Main API endpoints
        </div>
        <div class="endpoint">
            <code>🔌 /ws</code> - WebSocket connection
        </div>
        <div class="endpoint">
            <code>🗺️ /routes</code> - List all routes (debug)
        </div>
        
        <h2>🚀 Quick Start</h2>
        <p>Visit <a href="/docs">/docs</a> for interactive API documentation or use the WebSocket endpoint for real-time predictions.</p>
        
        <hr>
        <p style="color: #64748b; font-size: 0.9em;">Earthquake Prediction Service - Running on FastAPI</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)