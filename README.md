# Wrappuccino - FastAPI Model Deployment Wrapper

This repository provides a simple, structured way for data scientists to deploy their machine learning models as REST APIs using FastAPI, Gunicorn, and Docker.

## Features
- **Plug-and-play model deployment**: Just drop your `.pkl` models into the `models/` folder.
- **Automatic API generation**: FastAPI automatically creates interactive API docs at `/docs`.
- **Scalability**: Uses Gunicorn + Uvicorn workers for better performance.
- **Customizable Preprocessing**: Users can modify `preprocessing.py` or create a custom preprocessing script.
- **Containerized deployment**: Easily deploy using Docker.

---

## 📁 Folder Structure
```
model_api_wrapper/
│── models/              # Folder where users place their models (.pkl files)
│   ├── text_model.pkl
│   ├── text_vectorizer.pkl
│── app/                 # FastAPI application
│   ├── main.py          # API entry point
│   ├── model_loader.py  # Handles dynamic model loading
│   ├── pipeline.py      # Full ETL + inference pipeline
│   ├── preprocessing_template.py  # Template for custom preprocessing
│   ├── schemas.py       # Defines request/response schemas
│── requirements.txt      # Dependencies (FastAPI, Gunicorn, etc.)
│── Dockerfile           # Containerization setup
│── README.md            # Documentation (this file)
```

---

## How to Use

### 1️⃣ Install Dependencies (For Local Development)
```sh
pip install -r requirements.txt
```

### 2️⃣ Add Your Model
- Place your **Pickle (.pkl) model files** inside the `models/` directory.
- Ensure the models are trained using **scikit-learn** or a compatible framework.
- Model names will be inferred from the filenames (e.g., `text_model.pkl` → `text_model`).

### 3️⃣ Customize Preprocessing (Optional)
- Use the provided **preprocessing template** (`preprocessing_template.py`).
- Copy and rename it to `app/my_preprocessing.py`:
  ```sh
  cp app/preprocessing_template.py app/my_preprocessing.py
  ```
- Modify `custom_preprocess()` inside `my_preprocessing.py` to fit your model’s needs.
- This script will run **before vectorization and model inference**.

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
```json
{
  "available_models": ["text_model", "another_model"]
}
```

#### **Call from Command Line**
```sh
curl -X 'GET' 'http://localhost:8000/models' -H 'accept: application/json'
```

#### **Call from Python (requests library)**
```python
import requests

response = requests.get("http://localhost:8000/models")
print(response.json())
```

### **2️⃣ Make a Prediction (Using a Specific Model)**
**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "model_name": "text_model",
  "features": [0.12, 0.34, 0.56]
}
```

#### **Call from Command Line**
```sh
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "text_model", "features": [0.12, 0.34, 0.56]}'
```

#### **Call from Python (requests library)**
```python
import requests
import json

data = {
    "model_name": "text_model",
    "features": [0.12, 0.34, 0.56]
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### **3️⃣ Make a Prediction (Using the Full Pipeline)**
**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "use_pipeline": true,
  "text": "This is an example sentence for classification."
}
```

#### **Call from Command Line**
```sh
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"use_pipeline": true, "text": "This is an example sentence for classification."}'
```

#### **Call from Python (requests library)**
```python
import requests
import json

data = {
    "use_pipeline": True,
    "text": "This is an example sentence for classification."
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

---

## Troubleshooting
**Model Not Found?**
- Ensure the `.pkl` file is inside the `models/` folder.
- Check that the model name in the request matches the filename.

**Custom Preprocessing Not Applied?**
- Ensure your custom preprocessing script is copied and renamed to `app/my_preprocessing.py`.
- Make sure `pipeline.py` is correctly set to use `my_preprocessing.py`.

**Docker Port Issue?**
- Make sure port `8000` is available.
- Try changing the port mapping when running Docker (`-p 8080:8000`).

---

## License
This project is licensed under the MIT License.


