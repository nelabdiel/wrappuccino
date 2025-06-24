# **Wrappuccino** — FastAPI ML Pipeline Wrapper

Wrappuccino provides a clean, modular way to deploy full machine learning pipelines as REST APIs using FastAPI and Gunicorn. Each pipeline can consist of a preprocessing script, a vectorizer, and a model—all packaged into a single folder for clarity and reusability.

---

## 🚀 Features

* **Pipeline-based organization**: Each ML pipeline lives in its own subfolder under `pipelines/`
* **Optional preprocessing**: Supports modular text transformations before vectorization
* **Automatic API generation**: REST API with comprehensive endpoint documentation
* **Scalable**: FastAPI with Gunicorn for production deployment
* **Easy to extend**: Add new pipelines by simply creating folders with model files

---

## 📁 Project Structure

```
wrappuccino/
├── pipelines/
│   ├── sentiment_classification/
│   │   ├── preprocessing.py       # Custom text preprocessing
│   │   ├── vectorizer.pkl         # TF-IDF vectorizer
│   │   └── model.pkl              # Random Forest classifier
│   ├── iris_classifier/
│   │   └── model.pkl              # Iris dataset classifier
├── app.py                         # Main FastAPI application
├── main.py                        # Application entry point
├── model_loader.py                # ML model loading utilities
├── pipeline.py                    # Pipeline discovery and validation
└── README.md                      # This file
```

---

## ⚙️ Quick Start

### 1️⃣ Install Dependencies

The project uses `uv` for dependency management. Dependencies are automatically installed:

- FastAPI
- uvicorn
- scikit-learn
- numpy
- pydantic
- requests

### 2️⃣ Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

Or alternatively:
```bash
python app.py
```

### 3️⃣ Access the API

The server will start on `http://localhost:5000` with the following endpoints:

- **API Documentation**: `http://localhost:5000/docs` (Interactive Swagger UI)
- **ReDoc Documentation**: `http://localhost:5000/redoc` (Alternative documentation)
- **Health Check**: `http://localhost:5000/health`
- **List Pipelines**: `http://localhost:5000/pipelines`
- **Make Predictions**: `http://localhost:5000/predict`

## 📊 API Usage Examples

---

## 🔌 API Endpoints

### 📦 `GET /pipelines`

Returns a list of available pipeline folders.

**Example Response:**
```json
{
  "available_pipelines": ["iris_classifier", "sentiment_classification"]
}
```

### 🤖 `POST /predict`

Use this endpoint to run predictions via ML pipelines.

#### Request Body for Text Pipeline:

```json
{
  "pipeline_name": "sentiment_classification",
  "text": "I love this product! It works perfectly."
}
```

#### Request Body for Numeric Pipeline:

```json
{
  "pipeline_name": "iris_classifier",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

#### Example Response:

```json
{
  "pipeline_name": "sentiment_classification",
  "prediction": 1,
  "confidence": 0.93,
  "preprocessing_applied": true,
  "vectorizer_applied": true
}
```

### 🏥 `GET /health`

Health check endpoint for monitoring.

**Example Response:**
```json
{
  "status": "healthy",
  "pipelines_loaded": 2
}
```

---

## 🔧 Adding New Pipelines

To add a new ML pipeline, create a folder under `pipelines/` with the following structure:

```
pipelines/your_pipeline_name/
├── model.pkl              # Required: Your trained ML model
├── vectorizer.pkl         # Optional: For text processing
└── preprocessing.py       # Optional: Custom preprocessing functions
```


### PyTorch Pipeline:
```
pipelines/your_pytorch_pipeline/
├── model.pth              # Required: PyTorch model (.pth or .pt)
├── model_architecture.py  # Required if using state dict
├── label_encoder.pkl      # Optional: Label mapping
├── vectorizer.pkl         # Optional: For text processing
└── preprocessing.py       # Optional: Custom preprocessing functions
```

### ONNX Pipeline:
```
pipelines/your_onnx_pipeline/
├── model.onnx             # Required: ONNX model file
├── label_encoder.pkl      # Optional: Label mapping
├── vectorizer.pkl         # Optional: For text processing
└── preprocessing.py       # Optional: Custom preprocessing functions
```

### Pipeline Components

#### Model Files (Required - One of these):
1. **`model.pkl`** - Scikit-learn model saved with pickle/joblib
2. **`model.pth` or `model.pt`** - PyTorch model (full model or state dict)
3. **`model.onnx`** - ONNX optimized model for cross-platform deployment

#### Optional Components:
4. **`model_architecture.py`** (PyTorch only)
   - Required if saving PyTorch state dict instead of full model
   - Must define `create_model()` function that returns model instance

5. **`label_encoder.pkl`** (PyTorch/ONNX)
   - Maps numeric model outputs to meaningful labels
   - Used for classification tasks with string labels

6. **`vectorizer.pkl`** (All model types)
   - Text vectorizer (TF-IDF, CountVectorizer, etc.)
   - Converts text to numerical features for the model

7. **`preprocessing.py`** (All model types)
   - Must define a `custom_preprocess(text: str) -> str` function
   - Applied before vectorization for domain-specific text cleaning


### Example Pipeline Creation

```python
# Create and save a model
from sklearn.ensemble import RandomForestClassifier
import pickle

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open('pipelines/my_pipeline/model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

The pipeline will be automatically discovered and available via the API.

---

## 🧪 Testing the API

### Using curl

```bash
# Test root endpoint
curl http://localhost:5000/

# List available pipelines
curl http://localhost:5000/pipelines

# Make a prediction with numeric features
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "iris_classifier", "features": [5.1, 3.5, 1.4, 0.2]}'

# Make a prediction with text
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "sentiment_classification", "text": "I love this product!"}'
```

### Using Python

```python
import requests

# Test numeric prediction
iris_data = {
    "pipeline_name": "iris_classifier",
    "features": [5.1, 3.5, 1.4, 0.2]
}
response = requests.post("http://localhost:5000/predict", json=iris_data)
print(response.json())

# Test text prediction
text_data = {
    "pipeline_name": "sentiment_classification",
    "text": "Today's working was incredible. I couldn't be happier!"
}
response = requests.post("http://localhost:5000/predict", json=text_data)
print(response.json())
```

---

## 📊 Sample Pipelines

### Iris Classifier
- **Type**: Numeric features
- **Input**: 4 numeric features (sepal length, sepal width, petal length, petal width)
- **Output**: Species classification (0=setosa, 1=versicolor, 2=virginica)
- **Model**: Random Forest Classifier

### Sentiment Classification
- **Type**: Text processing
- **Input**: Text expressing opinions or sentiments
- **Output**: Binary classification (0=negative sentiment, 1=positive sentiment)
- **Features**: Custom preprocessing + TF-IDF vectorization + Random Forest
- **Preprocessing**: Text cleaning, lowercasing, special character removal

---

## 🚀 Production Deployment

For production deployment, use Gunicorn with proper configuration:

```bash
gunicorn --bind 0.0.0.0:5000 --worker-class sync --workers 4 main:app
```

### Environment Variables

- `SESSION_SECRET`: Flask session secret key (defaults to "wrappuccino-secret-key")

---

## 🔍 Troubleshooting

### Common Issues

1. **Pipeline not found**
   - Ensure `model.pkl` exists in the pipeline folder
   - Check that the folder name matches the `pipeline_name` in requests

2. **Preprocessing errors**
   - Verify `preprocessing.py` defines `custom_preprocess(text: str) -> str`
   - Check for import errors in the preprocessing module

3. **Vectorizer issues**
   - Ensure vectorizer is trained and saved properly
   - Verify it implements the `.transform()` method

4. **Model prediction errors**
   - Check that input features match the model's expected format
   - Ensure the model was trained and saved correctly

### Logs

Server logs provide detailed information about:
- Pipeline discovery and validation
- Model loading success/failure
- Prediction request processing
- Error details and stack traces

---

## 🎯 API Response Format

All prediction responses follow this format:

```json
{
  "pipeline_name": "string",           // Name of the pipeline used
  "prediction": "any",                 // Model prediction result
  "confidence": 0.95,                  // Confidence score (0-1, optional)
  "preprocessing_applied": true,       // Whether preprocessing was used
  "vectorizer_applied": true          // Whether vectorization was applied
}
```

Error responses:

```json
{
  "error": "string"                    // Error description
}
```

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

To contribute to Wrappuccino:

1. Fork the repository
2. Create your feature branch
3. Add your ML pipeline following the structure guidelines
4. Test your pipeline with the API
5. Submit a pull request

---

**Happy ML serving with Wrappuccino! ☕🤖**