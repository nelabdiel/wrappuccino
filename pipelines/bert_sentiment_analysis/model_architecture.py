"""
Model architecture script for BERT sentiment classification.
This script defines how to create and load a BERT model for the pipeline.
"""
import torch
import pickle
from pathlib import Path
try:
    from transformers import BertForSequenceClassification, BertTokenizer
except ImportError:
    raise ImportError("transformers library required for BERT models. Install with: pip install transformers")

def create_model():
    """
    Create and return a BERT model instance ready for state dict loading.
    This function will be called by the model loader.
    """
    
    
    # Get the pipeline directory
    pipeline_dir = Path(__file__).parent
    
    # Load label encoder to determine number of classes
    label_encoder_path = pipeline_dir / "label_encoder.pkl"
    if label_encoder_path.exists():
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        num_labels = len(label_encoder.classes_)
    else:
        # Default to binary classification if no label encoder
        num_labels = 2
    
    # Create BERT model with the correct number of labels
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    return model

def get_tokenizer():
    """
    Get the BERT tokenizer for this model.
    This will be used during preprocessing.
    """
    
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer

def preprocess_for_bert(text, max_length=512):
    """
    Preprocess text specifically for BERT input.
    This replaces the standard preprocessing when BERT is detected.
    """
    tokenizer = get_tokenizer()
    
    # Tokenize and encode the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    return inputs

def bert_predict(model, text):
    """
    Make prediction using BERT model with proper tokenization.
    """
    # Tokenize the input
    inputs = preprocess_for_bert(text)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get prediction and confidence
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence
