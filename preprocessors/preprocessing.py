import re
import spacy

# Load Spacy model (or use NLTK)
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """Applies text cleaning, lowercasing, lemmatization, and regex filtering."""
    
    # Lowercasing
    text = text.lower()

    # Remove special characters & digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lemmatization (using Spacy)
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    return lemmatized_text
