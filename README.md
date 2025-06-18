# **Wrappuccino** — FastAPI ML Pipeline Wrapper

Wrappuccino provides a clean, modular way to deploy full machine learning pipelines as REST APIs using FastAPI and Gunicorn. Each pipeline can consist of a preprocessing script, a vectorizer, and a model (scikit-learn, PyTorch, or ONNX)—all packaged into a single folder for clarity and reusability.

---

## Features

* **Multi-framework support**: Deploy scikit-learn (.pkl), PyTorch (.pth/.pt), and ONNX (.onnx) models
* **Pipeline-based organization**: Each ML pipeline lives in its own subfolder under `pipelines/`
* **Optional preprocessing**: Supports modular text transformations before vectorization
* **Automatic API generation**: REST API with comprehensive endpoint documentation
* **Scalable**: FastAPI with Gunicorn for production deployment
* **Easy to extend**: Add new pipelines by simply creating folders with model files

---

## Project Structure

```
wrappuccino/
├── pipelines/
│   ├── sentiment_classification/     # Scikit-learn pipeline
│   │   ├── preprocessing.py          # Custom text preprocessing
│   │   ├── vectorizer.pkl            # TF-IDF vectorizer
│   │   └── model.pkl                 # Random Forest classifier
│   ├── pytorch_sentiment/            # PyTorch pipeline
│   │   ├── preprocessing.py          # Custom text preprocessing
│   │   ├── vectorizer.pkl            # TF-IDF vectorizer
│   │   ├── model.pth                 # PyTorch neural network
│   │   ├── model_architecture.py     # Model class definition
│   │   └── label_encoder.pkl         # Label mapping
│   ├── onnx_sentiment/               # ONNX pipeline
│   │   ├── preprocessing.py          # Custom text preprocessing
│   │   ├── vectorizer.pkl            # TF-IDF vectorizer
│   │   ├── model.onnx                # ONNX optimized model
│   │   └── label_encoder.pkl         # Label mapping
│   └── iris_classifier/              # Simple numeric pipeline
│       └── model.pkl                 # Iris dataset classifier
├── app.py                            # Main FastAPI application
├── main.py                           # Application entry point
├── model_loader.py                   # ML model loading utilities
├── pipeline.py                       # Pipeline discovery and validation
└── README.md                         # This file
```

---

## Quick Start

### 1) Install Dependencies

Core dependencies are automatically installed:

- FastAPI
- uvicorn
- scikit-learn
- numpy
- pydantic
- requests
- joblib

**For PyTorch models (.pth/.pt):**
```bash
pip install torch torchvision
```

**For ONNX models (.onnx):**
```bash
pip install onnx onnxruntime
```

**Optional conversion tools:**
```bash
pip install onnxmltools skl2onnx  # Convert sklearn to ONNX
```

### 2) Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

Or alternatively:
```bash
python app.py
```

### 3) Access the API

The server will start on `http://localhost:5000` with the following endpoints:

- **API Documentation**: `http://localhost:5000/docs` (Interactive Swagger UI)
- **ReDoc Documentation**: `http://localhost:5000/redoc` (Alternative documentation)
- **Health Check**: `http://localhost:5000/health`
- **List Pipelines**: `http://localhost:5000/pipelines`
- **Make Predictions**: `http://localhost:5000/predict`

## API Usage Examples

---

## API Endpoints

### `GET /pipelines`

Returns a list of available pipeline folders.

**Example Response:**
```json
{
  "available_pipelines": ["iris_classifier", "sentiment_classification"]
}
```

### `POST /predict`

Use this endpoint to run predictions via ML pipelines.

#### Request Body for Text Pipeline:

```json
{
  "pipeline_name": "sentiment_classification",
  "text": "Today's working was incredible. I couldn't be happier!"
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

### `GET /health`

Health check endpoint for monitoring.

**Example Response:**
```json
{
  "status": "healthy",
  "pipelines_loaded": 2
}
```

---

## Adding New Pipelines

To add a new ML pipeline, create a folder under `pipelines/` with the following structure:

### Scikit-learn Pipeline:
```
pipelines/your_sklearn_pipeline/
├── model.pkl              # Required: Trained sklearn model
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

#### Scikit-learn Model:
```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train and save model
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'pipelines/my_sklearn_pipeline/model.pkl')
```

#### PyTorch Model:
```python
import torch
import torch.nn as nn

# Define model architecture
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Save model
model = SentimentClassifier(1000, 128, 2)
torch.save(model, 'pipelines/my_pytorch_pipeline/model.pth')
```

#### ONNX Conversion:
```python
import torch.onnx

# Convert PyTorch to ONNX
dummy_input = torch.randn(1, 1000)
torch.onnx.export(pytorch_model, dummy_input, 
                  'pipelines/my_onnx_pipeline/model.onnx',
                  input_names=['input'], output_names=['output'])
```

The pipeline will be automatically discovered and available via the API.

---

## Testing the API

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
  -d '{"pipeline_name": "sentiment_classification", "Today's working was incredible. I couldn't be happier!"}'
```

### Using Python

```python
import requests

# Test numeric prediction
iris_data = {
    "pipeline_name": "iris_classifier",
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Send request to the prediction endpoint
response = requests.post("http://localhost:5000/predict", json=iris_data)
print(response.json())

# Expected response
{
  "prediction": 0,
  "confidence": 0.981583497609436,
  "preprocessing_applied": false,
  "vectorizer_applied": false
}

# Define input text for a sentiment model
text_data = {
    "pipeline_name": "sentiment_classification",
    "text": "Today's working was incredible. I couldn't be happier!"
}

# Send request to the prediction endpoint
response = requests.post("http://localhost:5000/predict", json=text_data)
print(response.json())

# Expected response
{
  "prediction": 1,
  "confidence": 0.6177903691067675,
  "preprocessing_applied": true,
  "vectorizer_applied": true
}

```

---

## Sample Pipelines

### Iris Classifier (Scikit-learn)
- **Type**: Numeric features
- **Model**: `model.pkl` - Random Forest Classifier
- **Input**: 4 numeric features (sepal length, sepal width, petal length, petal width)
- **Output**: Species classification (0=setosa, 1=versicolor, 2=virginica)

### Sentiment Classification (Scikit-learn)
- **Type**: Text processing with full pipeline
- **Model**: `model.pkl` - Random Forest Classifier
- **Components**: `preprocessing.py` + `vectorizer.pkl` + `model.pkl`
- **Input**: Text expressing opinions or sentiments
- **Output**: Binary classification (0=negative sentiment, 1=positive sentiment)
- **Pipeline**: Text cleaning → TF-IDF vectorization → Classification

### PyTorch Sentiment Model (Example)
- **Type**: Deep learning text classification
- **Model**: `model.pth` - Neural Network
- **Components**: `preprocessing.py` + `vectorizer.pkl` + `model.pth` + `label_encoder.pkl`
- **Input**: Text data
- **Output**: Classified sentiment with confidence scores
- **Pipeline**: Text cleaning → TF-IDF → Neural network → Label decoding

### ONNX Optimized Model (Example)
- **Type**: Cross-platform optimized inference
- **Model**: `model.onnx` - Converted from PyTorch/TensorFlow
- **Components**: `preprocessing.py` + `vectorizer.pkl` + `model.onnx` + `label_encoder.pkl`
- **Input**: Any supported input format
- **Output**: Fast, optimized predictions
- **Pipeline**: Preprocessing → Vectorization → ONNX inference → Label mapping

---

## Universal Pipeline Architecture

**You're absolutely correct!** All model types (scikit-learn, PyTorch, ONNX) follow the same modular pipeline structure:

```
Common Pipeline Flow:
Text Input → preprocessing.py → vectorizer.pkl → model.{pkl|pth|onnx} → label_encoder.pkl → Final Output
```

### Pipeline Components Work Identically:

1. **`preprocessing.py`** - Same text cleaning function across all model types
2. **`vectorizer.pkl`** - Same TF-IDF/text vectorization for all models  
3. **`model.*`** - Different formats but same prediction interface
4. **`label_encoder.pkl`** - Same label mapping for PyTorch/ONNX models

### Example: Sentiment Analysis Across Frameworks

All three implementations use identical preprocessing and vectorization:

**Scikit-learn Pipeline:**
```
pipelines/sklearn_sentiment/
├── preprocessing.py      # Same cleaning function
├── vectorizer.pkl        # Same TF-IDF vectorizer
└── model.pkl            # RandomForest classifier
```

**PyTorch Pipeline:**
```
pipelines/pytorch_sentiment/
├── preprocessing.py      # Identical cleaning function
├── vectorizer.pkl        # Same TF-IDF vectorizer
├── model.pth            # Neural network
└── label_encoder.pkl    # Maps 0/1 to negative/positive
```

**ONNX Pipeline:**
```
pipelines/onnx_sentiment/
├── preprocessing.py      # Identical cleaning function
├── vectorizer.pkl        # Same TF-IDF vectorizer  
├── model.onnx           # Optimized version of any model
└── label_encoder.pkl    # Same label mapping
```

### Prediction Flow Example:

```python
# Same for all model types:
text = "I love this product!"

# 1. Preprocessing (identical)
clean_text = custom_preprocess(text)  # "i love this product"

# 2. Vectorization (identical) 
features = vectorizer.transform([clean_text])  # [0.2, 0.8, 0.1, ...]

# 3. Model prediction (framework-specific)
prediction = model.predict(features)  # 1

# 4. Label decoding (PyTorch/ONNX only)
label = label_encoder.inverse_transform([prediction])  # "positive"
```

This modular design allows you to:
- **Prototype** with scikit-learn
- **Upgrade** to PyTorch for better accuracy
- **Deploy** with ONNX for production performance
- **Reuse** preprocessing and vectorization components

---

## Production Deployment

For production deployment, use Gunicorn with proper configuration:

```bash
gunicorn --bind 0.0.0.0:5000 --worker-class sync --workers 4 main:app
```


---

## Troubleshooting

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

## API Response Format

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

## License

This project is open source and available under the MIT License.

---

## Contributing

To contribute to Wrappuccino:

1. Fork the repository
2. Create your feature branch
3. Add your ML pipeline following the structure guidelines
4. Test your pipeline with the API
5. Submit a pull request
