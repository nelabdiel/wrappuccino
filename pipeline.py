"""
Pipeline management module for discovering and validating ML pipelines.
Moved from app folder to root level for Flask implementation.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration class for individual ML pipelines."""
    
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.has_model = False
        self.has_vectorizer = False
        self.has_preprocessing = False
        self.has_label_encoder = False
        self.has_model_architecture = False
        self.model_path: Optional[Path] = None
        self.vectorizer_path: Optional[Path] = None
        self.preprocessing_path: Optional[Path] = None
        self.label_encoder_path: Optional[Path] = None
        self.model_architecture_path: Optional[Path] = None
        self.model_type: str = "sklearn"  # sklearn, pytorch, onnx
        
        self._validate_pipeline()
    
    def _validate_pipeline(self):
        """Validate pipeline components and set availability flags."""
        # Check for model files in order of preference: sklearn -> pytorch -> onnx
        model_files = [
            ("model.pkl", "sklearn"),
            ("model.pth", "pytorch"),
            ("model.pt", "pytorch"),
            ("model.onnx", "onnx")
        ]
        
        for filename, model_type in model_files:
            model_file = self.path / filename
            if model_file.exists():
                self.has_model = True
                self.model_path = model_file
                self.model_type = model_type
                break
        
        if not self.has_model:
            raise ValueError(f"Pipeline '{self.name}' missing required model file (.pkl, .pth, .pt, or .onnx)")
        
        # Check for optional vectorizer
        vectorizer_file = self.path / "vectorizer.pkl"
        if vectorizer_file.exists():
            self.has_vectorizer = True
            self.vectorizer_path = vectorizer_file
        
        # Check for optional preprocessing script
        preprocessing_file = self.path / "preprocessing.py"
        if preprocessing_file.exists():
            self.has_preprocessing = True
            self.preprocessing_path = preprocessing_file
        
        # Check for optional label encoder (for PyTorch/ONNX models)
        label_encoder_file = self.path / "label_encoder.pkl"
        if label_encoder_file.exists():
            self.has_label_encoder = True
            self.label_encoder_path = label_encoder_file
        
        # Check for optional model architecture (for complex PyTorch models)
        model_architecture_file = self.path / "model_architecture.py"
        if model_architecture_file.exists():
            self.has_model_architecture = True
            self.model_architecture_path = model_architecture_file
        
        logger.info(f"Pipeline '{self.name}' validated - Model: ✓, Vectorizer: {'✓' if self.has_vectorizer else '✗'}, Preprocessing: {'✓' if self.has_preprocessing else '✗'}, Architecture: {'✓' if self.has_model_architecture else '✗'}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline config to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "has_model": self.has_model,
            "has_vectorizer": self.has_vectorizer,
            "has_preprocessing": self.has_preprocessing,
            "has_label_encoder": self.has_label_encoder,
            "model_type": self.model_type,
            "model_path": str(self.model_path) if self.model_path else None,
            "vectorizer_path": str(self.vectorizer_path) if self.vectorizer_path else None,
            "preprocessing_path": str(self.preprocessing_path) if self.preprocessing_path else None,
            "label_encoder_path": str(self.label_encoder_path) if self.label_encoder_path else None,
            "model_architecture_path": str(self.model_architecture_path) if self.model_architecture_path else None
        }

class PipelineManager:
    """Manages discovery and validation of ML pipelines."""
    
    def __init__(self, pipelines_dir: str = "pipelines"):
        self.pipelines_dir = Path(pipelines_dir)
        self.available_pipelines: Dict[str, PipelineConfig] = {}
        
        # Create pipelines directory if it doesn't exist
        self.pipelines_dir.mkdir(exist_ok=True)
    
    def discover_pipelines(self) -> Dict[str, PipelineConfig]:
        """
        Discover and validate all available ML pipelines.
        
        Returns:
            Dict[str, PipelineConfig]: Dictionary of validated pipeline configurations
        """
        self.available_pipelines.clear()
        
        if not self.pipelines_dir.exists():
            logger.warning(f"Pipelines directory '{self.pipelines_dir}' does not exist")
            return self.available_pipelines
        
        # Scan for pipeline directories
        for item in self.pipelines_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    pipeline_config = PipelineConfig(item.name, item)
                    self.available_pipelines[item.name] = pipeline_config
                    logger.info(f"Successfully loaded pipeline: {item.name}")
                except Exception as e:
                    logger.error(f"Failed to load pipeline '{item.name}': {str(e)}")
        
        logger.info(f"Pipeline discovery complete. Found {len(self.available_pipelines)} valid pipelines")
        return self.available_pipelines
    
    def get_pipeline(self, name: str) -> Optional[PipelineConfig]:
        """
        Get a specific pipeline configuration.
        
        Args:
            name: Name of the pipeline
            
        Returns:
            PipelineConfig or None if not found
        """
        return self.available_pipelines.get(name)
    
    def list_pipelines(self) -> list[str]:
        """Get list of available pipeline names."""
        return list(self.available_pipelines.keys())
    
    def get_pipeline_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all pipelines."""
        return {
            name: config.to_dict() 
            for name, config in self.available_pipelines.items()
        }
