import re
import spacy

# Load Spacy model (or replace with other NLP tools like NLTK if needed)
nlp = spacy.load("en_core_web_sm")

def custom_preprocess(text: str) -> str:
    """
    Custom preprocessing function.
    Modify this function to fit your model's needs.
    Example transformations: lowercasing, lemmatization, regex filtering.
    """

    # Lowercasing
    text = text.lower()

    # Remove special characters & digits (modify regex as needed)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lemmatization (optional)
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    return lemmatized_text

# Example usage (for testing)
if __name__ == "__main__":
    sample_text = "Hello! This is a sample TEXT with numbers 123 and symbols!?"
    print("Original:", sample_text)
    print("Processed:", custom_preprocess(sample_text))
