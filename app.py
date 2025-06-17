"""
FastAPI-based ML Pipeline Wrapper - Wrappuccino
Provides REST API endpoints for ML model inference with dynamic pipeline loading.
"""

import logging
from typing import Dict, Any, List, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from pipeline import PipelineManager
from model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wrappuccino ML Pipeline Wrapper",
    description="Deploy ML models as REST APIs with modular pipeline organization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global pipeline manager
pipeline_manager = None
model_loaders = {}

# Pydantic models for request/response
class NumericPredictionRequest(BaseModel):
    pipeline_name: str = Field(..., description="Name of the ML pipeline to use")
    features: List[float] = Field(..., description="List of numeric features for prediction")

class TextPredictionRequest(BaseModel):
    pipeline_name: str = Field(..., description="Name of the ML pipeline to use")
    text: str = Field(..., description="Text input for prediction")

class PredictionResponse(BaseModel):
    pipeline_name: str
    prediction: Union[int, float, str]
    confidence: float
    preprocessing_applied: bool
    vectorizer_applied: bool

class PipelineInfo(BaseModel):
    available_pipelines: List[str]
    total_pipelines: int
    pipeline_details: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    pipelines_loaded: int
    available_pipelines: List[str]

def initialize_pipeline_manager():
    """Initialize pipeline manager."""
    global pipeline_manager
    if pipeline_manager is None:
        logger.info("Starting Wrappuccino ML Pipeline Wrapper")
        pipeline_manager = PipelineManager()
        pipelines = pipeline_manager.discover_pipelines()
        logger.info(f"Discovered {len(pipelines)} pipelines")

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline manager on startup."""
    initialize_pipeline_manager()

# Initialize immediately for WSGI compatibility
initialize_pipeline_manager()

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Wrappuccino ML Pipeline Wrapper",
        "version": "1.0.0",
        "description": "Deploy ML models as REST APIs with modular pipeline organization",
        "endpoints": {
            "pipelines": "/pipelines - List available pipelines",
            "predict": "/predict - Make predictions",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/pipelines", response_model=PipelineInfo)
async def list_pipelines():
    """Get a list of all available ML pipelines."""
    if not pipeline_manager:
        raise HTTPException(status_code=500, detail="Pipeline manager not initialized")
    
    pipeline_info = pipeline_manager.get_pipeline_info()
    pipeline_names = list(pipeline_info.keys())
    
    return PipelineInfo(
        available_pipelines=pipeline_names,
        total_pipelines=len(pipeline_names),
        pipeline_details=pipeline_info
    )

@app.post("/predict")
async def predict(request: dict):
    """Make a prediction using the specified ML pipeline."""
    if not pipeline_manager:
        raise HTTPException(status_code=500, detail="Pipeline manager not initialized")
    
    pipeline_name = request.get("pipeline_name")
    if not pipeline_name:
        raise HTTPException(status_code=400, detail="pipeline_name is required")
    
    pipeline_config = pipeline_manager.get_pipeline(pipeline_name)
    
    if not pipeline_config:
        available_pipelines = pipeline_manager.list_pipelines()
        raise HTTPException(
            status_code=404, 
            detail=f"Pipeline '{pipeline_name}' not found. Available pipelines: {available_pipelines}"
        )
    
    # Get or create model loader for this pipeline
    if pipeline_name not in model_loaders:
        model_loaders[pipeline_name] = ModelLoader(pipeline_config)
    
    model_loader = model_loaders[pipeline_name]
    
    try:
        # Determine prediction type based on request data
        if "text" in request:
            result = model_loader.predict_text(request["text"])
        elif "features" in request:
            result = model_loader.predict_features(request["features"])
        else:
            raise HTTPException(status_code=400, detail="Either 'text' or 'features' must be provided")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error for pipeline '{pipeline_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    if not pipeline_manager:
        return HealthResponse(
            status="unhealthy",
            pipelines_loaded=0,
            available_pipelines=[]
        )
    
    available_pipelines = pipeline_manager.list_pipelines()
    
    return HealthResponse(
        status="healthy",
        pipelines_loaded=len(available_pipelines),
        available_pipelines=available_pipelines
    )

@app.get("/docs-info")
async def docs_info():
    """API documentation information."""
    return {
        "title": "Wrappuccino ML Pipeline API",
        "description": "Complete API documentation for the ML Pipeline Wrapper",
        "interactive_docs": "/docs",
        "redoc_docs": "/redoc",
        "openapi_schema": "/openapi.json"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)