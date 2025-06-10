"""
FastAPI main application for The Projection Wizard API.
Provides REST endpoints for the ML pipeline functionality.
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.routes.endpoints import router as schema_router
from api.routes import pipeline

# Create FastAPI application
app = FastAPI(
    title="The Projection Wizard API",
    description="REST API for ML pipeline functionality",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(schema_router, prefix="/api/v1", tags=["schema"])
app.include_router(pipeline.router, tags=["pipeline"])


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
