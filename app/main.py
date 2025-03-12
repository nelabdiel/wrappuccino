from fastapi import FastAPI, HTTPException
from app.model_loader import models, load_models
from app.pipeline import run_pipeline
from app.schemas import PredictionInput

app = FastAPI()

# Load models on startup
load_models()

@app.post("/predict")
def predict(data: PredictionInput):
    if data.use_pipeline:
        try:
            prediction = run_pipeline(data.text)
            return {"pipeline": True, "prediction": prediction}
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # If calling a standalone model
    if data.model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[data.model_name]
    prediction = model.predict([data.features])
    return {"model": data.model_name, "prediction": prediction.tolist()}
