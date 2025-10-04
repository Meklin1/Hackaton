from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import inference
from src.api.routers import train

app = FastAPI(
    title="Exoplanet Classification Server",
    description="AI/ML server for exoplanet classification using NASA datasets",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(inference.router)
app.include_router(train.router)

# Health Endpoint
@app.get("/")
async def home():
    """Check Health status"""
    return {
        "message": "Exoplanet Classification Server is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "exoplanet-classification",
        "version": "1.0.0",
        "endpoints": {
            "data": "/data",
            "inference": "/inference",
            "docs": "/docs"
        }
    }