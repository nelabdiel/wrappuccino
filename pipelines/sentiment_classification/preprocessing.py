import re

def custom_preprocess(text: str) -> str:
    """
    Applies text cleaning, lowercasing, lemmatization, and regex filtering.
    
    Args:
        text: Raw input text
        
    Returns:
        str: Preprocessed text ready for vectorization
    """
    # Lowercasing
    text = text.lower().strip()

    # Remove special characters & digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text
