import joblib
import os
import importlib
from typing import Optional

MODEL_DIR = "models"
PREPROCESSOR_DIR = "preprocessors"
models = {}
vectorizers = {}

def load_pipeline():
    """Loads models and vectorizers from disk."""
    global models, vectorizers
    models.clear()
    vectorizers.clear()

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl"):
            name = filename[:-4]
            if "vectorizer" in name:
                vectorizers[name] = joblib.load(os.path.join(MODEL_DIR, filename))
            else:
                models[name] = joblib.load(os.path.join(MODEL_DIR, filename))

def list_preprocessors():
    """Lists available preprocessing scripts in preprocessors/"""
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
    prediction = model.predict(features)
    return prediction.tolist()
