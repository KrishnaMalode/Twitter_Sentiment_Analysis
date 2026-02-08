"""
NLP Pipeline: tokenization → stemming → lemmatization.
Used before TF-IDF vectorization (TF-IDF applied in sklearn).
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Downloads (idempotent)
def _ensure_nltk_data():
    for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


_ensure_nltk_data()

_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop = set(stopwords.words("english"))

def _clean(text):
    """Clean text but keep some realistic noise"""
    if not isinstance(text, str):
        return ""
    
    # Keep some hashtags/emojis for realism
    text = re.sub(r"http\S+|www\S+", "", text)  # Only remove URLs
    # text = re.sub(r"@\w+", "", text)         # COMMENTED: Keep @mentions
    # text = re.sub(r"#\w+", "", text)         # COMMENTED: Keep #hashtags
    
    text = re.sub(r"[^a-zA-Z0-9\s@#?!.,]+", " ", text)
    return " ".join(text.split())[:280]  # Twitter length limit



def tokenize(text):
    """Tokenize text (after cleaning)."""
    text = _clean(text)
    return word_tokenize(text) if text else []


def stem_tokens(tokens):
    """Apply Porter stemming to tokens."""
    return [_stemmer.stem(t) for t in tokens if t]


def lemmatize_tokens(tokens):
    """Apply WordNet lemmatization (verb by default)."""
    return [_lemmatizer.lemmatize(t, pos="v") for t in tokens if t]


def pipeline(text, remove_stopwords=True):
    """
    Full pipeline: tokenization → stemming → lemmatization.
    Returns space-joined string for TF-IDF input.
    """
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in _stop]
    tokens = stem_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def pipeline_for_vectorizer(text):
    """Wrapper for sklearn TfidfVectorizer: returns processed string."""
    return pipeline(text, remove_stopwords=True)

