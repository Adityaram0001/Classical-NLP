import re
import string

def clean_text_basic(text):
    """
    Basic text cleaning: lowercases, removes punctuation, removes extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers if needed (can be optional depending on task)
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_stopwords(tokens, stop_words_set):
    """
    Removes stopwords from a list of tokens.
    """
    return [t for t in tokens if t not in stop_words_set]
