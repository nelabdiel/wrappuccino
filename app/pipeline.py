import os
import joblib
import importlib
import torch
import numpy as np
from typing import Optional

MODEL_DIR = "models"
PREPROCESSOR_DIR = "preprocessors"
models = {}
vectorizers = {}

def load_model_file(filepath):
    if filepath.endswith(".pkl"):
        return joblib.load(filepath)
    elif filepath.endswith(".pth"):
        model = torch.load(filepath, map_location=torch.device("cpu"))
        if isinstance(model, torch.nn.Module):
            model.eval()
        return model
    else:
        raise ValueError(f"Unsupported model format: {filepath}")

def load_pipeline():
    """Loads models and vectorizers from disk."""
    global models, vectorizers
    models.clear()
    vectorizers.clear()

    for filename in os.listdir(MODEL_DIR):
        filepath = os.path.join(MODEL_DIR, filename)
        if filename.endswith(".pkl") or filename.endswith(".pth"):
            name = filename.rsplit(".", 1)[0]
            if "vectorizer" in name:
                vectorizers[name] = load_model_file(filepath)
            else:
                models[name] = load_model_file(filepath)

def list_preprocessors():
    """Lists available preprocessing scripts in preprocessors"""
    return [
        f[:-3] for f in os.listdir(PREPROCESSOR_DIR)
        if f.endswith(".py") and not f.startswith("__")
    ]

def run_pipeline(text: str, model_name: str, preprocessing_pipeline: Optional[str] = None):
    """Runs preprocessing → vectorizer → model prediction"""
    
    # Optional preprocessing
    if preprocessing_pipeline:
        try:
            module_path = f"preprocessors.{preprocessing_pipeline}"
            module = importlib.import_module(module_path)
            text = module.custom_preprocess(text)
        except ModuleNotFoundError:
            raise ValueError(f"Preprocessing script '{preprocessing_pipeline}' not found.")
        except AttributeError:
            raise ValueError(f"'{preprocessing_pipeline}' must define a 'custom_preprocess(text)' function.")

    # Vectorizer
    vec = vectorizers.get("text_vectorizer")
    if not vec:
        raise ValueError("Vectorizer not found.")
    features = vec.transform([text]).toarray()

    # Model
    model = models.get(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found.")

    if hasattr(model, "predict"):  # scikit-learn
        prediction = model.predict(features)
        return prediction.tolist()

    elif isinstance(model, torch.nn.Module):  # PyTorch
        input_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
        if output.shape[1] == 1:
            return output.squeeze().tolist()
        return output.argmax(dim=1).tolist()

    else:
        raise ValueError(f"Model type not supported for: {model_name}")
