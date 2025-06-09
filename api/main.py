"""
FastAPI main application for The Projection Wizard API.
Provides REST endpoints for the ML pipeline functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import route modules
from api.routes.endpoints import router as schema_router

# Create FastAPI application
app = FastAPI(
    title="The Projection Wizard API",
    description="REST API for ML pipeline functionality",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(schema_router, prefix="/api/v1", tags=["schema"])

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "The Projection Wizard API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 