# Wrappuccino - FastAPI Model Deployment Wrapper

This repository provides a simple, structured way for data scientists to deploy their machine learning models as REST APIs using FastAPI, Gunicorn, and Docker.

## Features
- **Plug-and-play model deployment**: Just drop your `.pkl` or `.pth` models into the `models/` folder.
- **Automatic API generation**: FastAPI automatically creates interactive API docs at `/docs`.
- **Scalability**: Uses Gunicorn + Uvicorn workers for better performance.
- **Customizable Preprocessing**: Users can create modular preprocessing scripts and select them per request.
- **Containerized deployment**: Easily deploy using Docker.

---

## 📁 Folder Structure
```
model_api_wrapper/
│── models/                 # Folder where users place their models (.pkl or .pth files)
│   ├── text_model.pkl
│   ├── text_vectorizer.pkl
│   ├── iris_model.pkl      # <- Example test model
│   ├── pytorch_model.pth   # <- PyTorch model (optional)
│── preprocessors/          # Custom preprocessing scripts (user-defined)
│   ├── my_preprocessing.py
│   ├── regex_cleaner.py
│── app/                    # FastAPI application
│   ├── main.py             # API entry point
│   ├── model_loader.py     # Handles dynamic model loading
│   ├── pipeline.py         # Full ETL + inference pipeline
│   ├── schemas.py          # Defines request/response schemas
│── requirements.txt        # Dependencies (FastAPI, Gunicorn, etc.)
│── Dockerfile              # Containerization setup
│── README.md               # Documentation (this file)
```

---

## How to Use

### 1️⃣ Install Dependencies (For Local Development)
```sh
pip install -r requirements.txt
```

> If using PyTorch models, make sure `torch` is included in `requirements.txt`:
> ```txt
> torch
> ```

### 2️⃣ Add Your Model
- Place your **Pickle (.pkl)** or **PyTorch (.pth)** model files inside the `models/` directory.
- `.pkl` files should be trained with **scikit-learn** or compatible.
- `.pth` files must contain a full `torch.nn.Module`, not just a `state_dict()`.

### 3️⃣ Add a Preprocessing Script (Optional)
- Create your preprocessing script inside the `preprocessors/` folder.
- Define a `custom_preprocess(text: str) -> str` function in the script.
- Example: `preprocessors/my_preprocessing.py`

### 4️⃣ Run the API (Locally)
```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 5️⃣ Run with Gunicorn (Production-Ready)
```sh
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
```

### 6️⃣ Deploy with Docker

#### Build the Docker Image
```sh
docker build -t fastapi-model-api .
```

#### Run the Container
```sh
docker run -p 8000:8000 -v $(pwd)/models:/app/models fastapi-model-api
```

---

## API Endpoints

### **1️⃣ List Available Models**
**Endpoint:** `GET /models`

#### Example Response:
```json
{
  "available_models": ["text_model", "iris_model", "pytorch_model"]
}
```

#### Command Line:
```sh
curl -X GET "http://localhost:8000/models" -H "accept: application/json"
```

#### Python:
```python
import requests

response = requests.get("http://localhost:8000/models")
print(response.json())
```

---

### **2️⃣ List Available Preprocessing Pipelines**
**Endpoint:** `GET /preprocessors`

#### Example Response:
```json
{
  "available_preprocessors": ["my_preprocessing", "regex_cleaner"]
}
```

#### Command Line:
```sh
curl -X GET "http://localhost:8000/preprocessors" -H "accept: application/json"
```

#### Python:
```python
import requests

response = requests.get("http://localhost:8000/preprocessors")
print(response.json())
```

---

### **3️⃣ Make a Prediction (Using a Specific Model)**
**Endpoint:** `POST /predict`

#### ✅ Example: `iris_model` without preprocessing
```json
{
  "model_name": "iris_model",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Command Line:**
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "iris_model", "features": [5.1, 3.5, 1.4, 0.2]}'
```

**Python:**
```python
import requests

data = {
    "model_name": "iris_model",
    "features": [5.1, 3.5, 1.4, 0.2]
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

---

#### ✅ Example: `text_model` with preprocessing pipeline
```json
{
  "model_name": "text_model",
  "preprocessing_pipeline": "my_preprocessing",
  "use_pipeline": true,
  "text": "This is an example sentence."
}
```

**Command Line:**
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "text_model", "preprocessing_pipeline": "my_preprocessing", "use_pipeline": true, "text": "This is an example sentence."}'
```

**Python:**
```python
import requests

data = {
    "model_name": "text_model",
    "preprocessing_pipeline": "my_preprocessing",
    "use_pipeline": True,
    "text": "This is an example sentence."
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

---

## Troubleshooting

**Model Not Found?**
- Ensure the model file is inside the `models/` folder.
- Make sure the model name in the request matches the file name (without extension).

**PyTorch Model Not Working?**
- Ensure the `.pth` file contains a full `torch.nn.Module` object, not just a `state_dict()`.
- If you're using a `state_dict()`, you'll need a script to reconstruct and load the model manually.

**Preprocessing Script Not Found?**
- Ensure your script is inside `preprocessors/` and named accordingly.
- It must contain a `custom_preprocess(text: str)` function.

**Docker Port Issue?**
- Make sure port `8000` is available.
- Try changing the port mapping when running Docker (`-p 8080:8000`).

---

## License

This project is licensed under the MIT License.

