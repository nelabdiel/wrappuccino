import joblib
import os

MODEL_DIR = "models"
models = {}

def load_models():
    """Loads all .pkl models from the models directory."""
    global models
    models.clear()
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl"):
            model_name = filename[:-4]  # Remove .pkl extension
            models[model_name] = joblib.load(os.path.join(MODEL_DIR, filename))
