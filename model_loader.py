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
import onnxruntime as ort

from pipeline import PipelineConfig

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and inference for ML pipeline components."""
    
    def __init__(self, pipeline_config: PipelineConfig):
        self.config = pipeline_config
        self.model = None
        self.vectorizer = None
        self.preprocessing_func = None
        self.label_encoder = None
        
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
            
            # Load optional label encoder
            if self.config.has_label_encoder:
                self._load_label_encoder()
                
        except Exception as e:
            logger.error(f"Error loading pipeline components: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the ML model based on model type (sklearn, pytorch, onnx)."""
        if self.config.model_type == "sklearn":
            self._load_sklearn_model()
        elif self.config.model_type == "pytorch":
            self._load_pytorch_model()
        elif self.config.model_type == "onnx":
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _load_sklearn_model(self):
        """Load scikit-learn model from pickle file."""
        try:
            with open(self.config.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Scikit-learn model loaded successfully from {self.config.model_path}")
        except Exception as e:
            logger.warning(f"Standard pickle loading failed: {str(e)}")
            # Try loading with joblib for better scikit-learn compatibility
            try:
                import joblib
                self.model = joblib.load(self.config.model_path)
                logger.info(f"Scikit-learn model loaded successfully with joblib from {self.config.model_path}")
            except ImportError:
                logger.error("joblib not available for model loading")
                raise ValueError(f"Failed to load model - joblib required: {str(e)}")
            except Exception as joblib_error:
                logger.error(f"Failed to load scikit-learn model with joblib: {str(joblib_error)}")
                raise ValueError(f"Model loading failed - incompatible format: {str(joblib_error)}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model from .pth or .pt file."""
        try:
            import torch
            # Load the model state dict
            device = torch.device('cpu')  # Use CPU for inference
            self.model = torch.load(self.config.model_path, map_location=device)
            
            # If it's a state dict, we need the model architecture defined elsewhere
            if isinstance(self.model, dict):
                logger.warning("Loaded PyTorch state dict - model architecture must be defined in model_architecture.py")
                # Try to load model architecture
                arch_file = self.config.model_path.parent / "model_architecture.py"
                if arch_file.exists():
                    spec = importlib.util.spec_from_file_location("model_architecture", arch_file)
                    arch_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(arch_module)
                    
                    if hasattr(arch_module, 'create_model'):
                        model_instance = arch_module.create_model()
                        model_instance.load_state_dict(self.model)
                        self.model = model_instance
                    else:
                        raise ValueError("model_architecture.py must define a 'create_model()' function")
                else:
                    raise ValueError("PyTorch state dict requires model_architecture.py file")
            
            self.model.eval()  # Set to evaluation mode
            logger.info(f"PyTorch model loaded successfully from {self.config.model_path}")
            
        except ImportError:
            raise ValueError("PyTorch not installed - required for .pth/.pt models")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise ValueError(f"PyTorch model loading failed: {str(e)}")
    
    def _load_onnx_model(self):
        """Load ONNX model from .onnx file."""
        try:
            
            # Create ONNX Runtime inference session
            self.model = ort.InferenceSession(str(self.config.model_path))
            
            # Get input and output info
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            
            logger.info(f"ONNX model loaded successfully from {self.config.model_path}")
            logger.info(f"Input: {self.input_name}, Output: {self.output_name}")
            
        except ImportError:
            raise ValueError("onnxruntime not installed - required for .onnx models")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise ValueError(f"ONNX model loading failed: {str(e)}")
    
    def _load_vectorizer(self):
        """Load the vectorizer from pickle file."""
        try:
            with open(self.config.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"Vectorizer loaded successfully from {self.config.vectorizer_path}")
        except Exception as e:
            logger.warning(f"Standard pickle loading failed for vectorizer: {str(e)}")
            # Try loading with joblib for better scikit-learn compatibility
            try:
                import joblib
                self.vectorizer = joblib.load(self.config.vectorizer_path)
                logger.info(f"Vectorizer loaded successfully with joblib from {self.config.vectorizer_path}")
            except Exception as joblib_error:
                logger.error(f"Failed to load vectorizer with joblib: {str(joblib_error)}")
                raise ValueError(f"Vectorizer loading failed: {str(joblib_error)}")
    
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
    
    def _load_label_encoder(self):
        """Load label encoder from pickle file."""
        try:
            with open(self.config.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded successfully from {self.config.label_encoder_path}")
        except Exception as e:
            logger.warning(f"Standard pickle loading failed for label encoder: {str(e)}")
            # Try loading with joblib
            try:
                import joblib
                self.label_encoder = joblib.load(self.config.label_encoder_path)
                logger.info(f"Label encoder loaded successfully with joblib from {self.config.label_encoder_path}")
            except Exception as joblib_error:
                logger.error(f"Failed to load label encoder with joblib: {str(joblib_error)}")
                raise ValueError(f"Label encoder loading failed: {str(joblib_error)}")
    
    def _make_prediction(self, features):
        """Make prediction based on model type."""
        if self.config.model_type == "sklearn":
            return self._predict_sklearn(features)
        elif self.config.model_type == "pytorch":
            return self._predict_pytorch(features)
        elif self.config.model_type == "onnx":
            return self._predict_onnx(features)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _predict_sklearn(self, features):
        """Make prediction with scikit-learn model."""
        prediction = self.model.predict(features)
        confidence = 1.0
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            confidence = float(probabilities[0].max())
        
        return prediction[0] if len(prediction) > 0 else prediction, confidence
    
    def _predict_pytorch(self, features):
        """Make prediction with PyTorch model."""
        try:
            import torch
            import numpy as np
            
            # Convert features to tensor
            if hasattr(features, 'toarray'):  # Sparse matrix
                features_array = features.toarray()
            else:
                features_array = np.array(features)
            
            input_tensor = torch.FloatTensor(features_array)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = outputs.numpy()
                prediction = np.argmax(probabilities, axis=1)[0]
                confidence = float(np.max(probabilities))
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"PyTorch prediction failed: {str(e)}")
            raise ValueError(f"PyTorch prediction error: {str(e)}")
    
    def _predict_onnx(self, features):
        """Make prediction with ONNX model."""
        try:
            import numpy as np
            
            # Convert features to numpy array
            if hasattr(features, 'toarray'):  # Sparse matrix
                features_array = features.toarray().astype(np.float32)
            else:
                features_array = np.array(features, dtype=np.float32)
            
            # Make prediction
            outputs = self.model.run([self.output_name], {self.input_name: features_array})
            probabilities = outputs[0]
            prediction = np.argmax(probabilities, axis=1)[0]
            confidence = float(np.max(probabilities))
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"ONNX prediction failed: {str(e)}")
            raise ValueError(f"ONNX prediction error: {str(e)}")
    
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
            
            # Make prediction based on model type
            prediction, confidence = self._make_prediction(features)
            
            # Apply label encoding if available
            if self.label_encoder and hasattr(self.label_encoder, 'inverse_transform'):
                try:
                    prediction = self.label_encoder.inverse_transform([prediction])[0]
                except:
                    pass  # Keep original prediction if label encoding fails
            
            # Ensure JSON serializable types
            import numpy as np
            if isinstance(prediction, (np.integer, np.int64, np.int32)):
                prediction = int(prediction)
            elif isinstance(prediction, (np.floating, np.float64, np.float32)):
                prediction = float(prediction)
            elif isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            if isinstance(confidence, (np.floating, np.float64, np.float32)):
                confidence = float(confidence)
            elif confidence is None:
                confidence = 1.0
            
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
