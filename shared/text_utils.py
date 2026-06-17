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

def apply_stemming(text):
    """
    Applies Porter Stemmer to the text.
    """
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in tokens])

def apply_lemmatization(text):
    """
    Applies WordNet Lemmatizer to the text.
    """
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])
