"""
Model loading and inference module for ML pipelines.
Handles loading of models, vectorizers, and preprocessing scripts.
"""

import pickle
import logging
import importlib.util
import sys
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
import numpy as np

from pipeline import PipelineConfig

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and inference for ML pipeline components."""
    
    def __init__(self, pipeline_config: PipelineConfig):
        self.config = pipeline_config
        self.model = None
        self.vectorizer = None
        self.preprocessing_func = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all pipeline components (model, vectorizer, preprocessing)."""
        try:
            # Load model (required)
            self._load_model()
            
            # Load optional vectorizer
            if self.config.has_vectorizer:
                self._load_vectorizer()
            
            # Load optional preprocessing
            if self.config.has_preprocessing:
                self._load_preprocessing()
                
        except Exception as e:
            logger.error(f"Error loading pipeline components: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the ML model from pickle file with compatibility handling."""
        try:
            with open(self.config.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.config.model_path}")
        except Exception as e:
            logger.warning(f"Standard pickle loading failed: {str(e)}")
            # Try loading with joblib for better scikit-learn compatibility
            try:
                import joblib
                self.model = joblib.load(self.config.model_path)
                logger.info(f"Model loaded successfully with joblib from {self.config.model_path}")
            except ImportError:
                logger.error("joblib not available for model loading")
                raise ValueError(f"Failed to load model - joblib required: {str(e)}")
            except Exception as joblib_error:
                logger.error(f"Failed to load model with joblib: {str(joblib_error)}")
                raise ValueError(f"Model loading failed - incompatible format: {str(joblib_error)}")
    
    def _load_vectorizer(self):
        """Load the vectorizer from pickle file."""
        try:
            with open(self.config.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"Vectorizer loaded successfully from {self.config.vectorizer_path}")
        except Exception as e:
            logger.error(f"Error loading vectorizer: {str(e)}")
            raise ValueError(f"Failed to load vectorizer: {str(e)}")
    
    def _load_preprocessing(self):
        """Load custom preprocessing function from Python module."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"{self.config.name}_preprocessing", 
                self.config.preprocessing_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for the custom_preprocess function
            if hasattr(module, 'custom_preprocess'):
                self.preprocessing_func = module.custom_preprocess
                logger.info(f"Preprocessing function loaded from {self.config.preprocessing_path}")
            else:
                logger.warning(f"No 'custom_preprocess' function found in {self.config.preprocessing_path}")
                
        except Exception as e:
            logger.error(f"Error loading preprocessing: {str(e)}")
            raise ValueError(f"Failed to load preprocessing: {str(e)}")
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on text input using the full pipeline.
        
        Args:
            text: Input text for prediction
            
        Returns:
            Dict containing prediction results and metadata
        """
        try:
            processed_text = text
            preprocessing_applied = False
            vectorizer_applied = False
            
            # Apply preprocessing if available
            if self.preprocessing_func:
                processed_text = self.preprocessing_func(text)
                preprocessing_applied = True
                logger.debug(f"Preprocessing applied: '{text}' -> '{processed_text}'")
            
            # Apply vectorization if available
            if self.vectorizer:
                features = self.vectorizer.transform([processed_text])
                vectorizer_applied = True
                logger.debug(f"Vectorization applied, shape: {features.shape}")
            else:
                # If no vectorizer, assume the model can handle text directly
                features = [processed_text]
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Get prediction confidence if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features)
                    confidence = float(np.max(proba))
                except Exception as e:
                    logger.debug(f"Could not get prediction probability: {str(e)}")
            
            # Format prediction result
            if isinstance(prediction, np.ndarray):
                if prediction.size == 1:
                    prediction = prediction.item()
                else:
                    prediction = prediction.tolist()
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "preprocessing_applied": preprocessing_applied,
                "vectorizer_applied": vectorizer_applied
            }
            
        except Exception as e:
            logger.error(f"Error during text prediction: {str(e)}")
            raise ValueError(f"Text prediction failed: {str(e)}")
    
    def predict_features(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction on numeric features.
        
        Args:
            features: List of numeric features
            
        Returns:
            Dict containing prediction results and metadata
        """
        try:
            # Convert to numpy array for prediction
            feature_array = np.array([features])
            
            # Make prediction
            prediction = self.model.predict(feature_array)
            
            # Get prediction confidence if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(feature_array)
                    confidence = float(np.max(proba))
                except Exception as e:
                    logger.debug(f"Could not get prediction probability: {str(e)}")
            
            # Format prediction result
            if isinstance(prediction, np.ndarray):
                if prediction.size == 1:
                    prediction = prediction.item()
                else:
                    prediction = prediction.tolist()
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "preprocessing_applied": False,
                "vectorizer_applied": False
            }
            
        except Exception as e:
            logger.error(f"Error during feature prediction: {str(e)}")
            raise ValueError(f"Feature prediction failed: {str(e)}")
