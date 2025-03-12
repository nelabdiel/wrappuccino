import joblib
import os
import importlib

# Load models and vectorizers
MODEL_DIR = "models"
models = {}
vectorizers = {}

# Default preprocessing module (users should rename their script to match this)
PREPROCESSING_MODULE = "app.my_preprocessing"

def load_pipeline():
    """Loads models, vectorizers, and the custom preprocessing script."""
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

    # Dynamically import the custom preprocessing module
    global preprocess_text
    try:
        preprocessing_module = importlib.import_module(PREPROCESSING_MODULE)
        preprocess_text = preprocessing_module.custom_preprocess
    except ModuleNotFoundError:
        raise ValueError(f"Custom preprocessing module '{PREPROCESSING_MODULE}' not found. Please rename your script.")

# Load everything on startup
load_pipeline()

def run_pipeline(text: str):
    """Runs the full ETL pipeline for text input."""
    # Preprocess the text using the custom function
    processed_text = preprocess_text(text)

    # Transform using vectorizer
    vec = vectorizers.get("text_vectorizer")
    if vec:
        features = vec.transform([processed_text]).toarray()
    else:
        raise ValueError("Vectorizer not found")

    # Predict using model
    model = models.get("text_model")
    if not model:
        raise ValueError("Model not found")

    prediction = model.predict(features)
    return prediction.tolist()
