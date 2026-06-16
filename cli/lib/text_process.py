from .search_utils import load_stopwords
from nltk.stem import PorterStemmer
import string

stemmer = PorterStemmer()

def normalize(text: str) -> str:
    return text.lower().translate(
        str.maketrans("","",string.punctuation)
    )

STOP_WORDS = load_stopwords()

def tokenize(text: str) -> list[str]:
    tokens = normalize(text).split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError("Term must tokenize to exactly one token")
    return tokens[0]