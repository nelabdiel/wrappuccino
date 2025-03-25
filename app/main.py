from fastapi import FastAPI, HTTPException
from app.model_loader import models, load_models
from app.pipeline import load_pipeline, run_pipeline, list_preprocessors
from app.schemas import PredictionInput

app = FastAPI()

# Load models and vectorizers on startup
load_models()
load_pipeline()

@app.get("/models")
def get_models():
    return {"available_models": list(models.keys())}

@app.get("/preprocessors")
def get_preprocessors():
    return {"available_preprocessors": list_preprocessors()}

@app.post("/predict")
def predict(data: PredictionInput):
    if data.use_pipeline:
        try:
            prediction = run_pipeline(
                text=data.text,
                model_name=data.model_name,
                preprocessing_pipeline=data.preprocessing_pipeline
            )
            return {"pipeline": True, "model": data.model_name, "prediction": prediction}
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # If not using pipeline, use raw feature input
    if data.model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[data.model_name]
    prediction = model.predict([data.features])
    return {"model": data.model_name, "prediction": prediction.tolist()}

