"""
FastAPI main application for The Projection Wizard API.
Provides REST endpoints for the ML pipeline functionality.
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Configure CORS origins from environment variable
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")

if allowed_origins_env == "*":
    # If the env var is exactly "*", use that for allow_origins
    configured_origins = ["*"]
elif allowed_origins_env:
    # Otherwise, parse it as a comma-separated list
    configured_origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]
else:
    # Fallback for local development if ALLOWED_ORIGINS env var is not set
    print("WARNING: ALLOWED_ORIGINS environment variable not set or empty. Using default dev origins.")
    configured_origins = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "https://lovable.dev/projects/dca90495-2ee9-4de4-86ef-b619f83fc331",
        "https://www.predictingwizard.com",  # Without trailing slash
        "https://www.predictingwizard.com/", # With trailing slash
    ]

print(f"INFO: Configuring CORS with origins: {configured_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
