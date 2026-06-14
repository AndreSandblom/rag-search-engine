import math
import os
import pickle
from collections import defaultdict, Counter

from .text_process import tokenize
from .search_utils import load_movies

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id,text):
        tokens = tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        # update per-document term frequency counts
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        doc = self.index.get(term,[])
        sorted_doc = sorted(doc)
        return sorted_doc

    def get_tf(self, doc_id, term):
        # Return frequency of `term` in document `doc_id` (0 if not present)
        return self.term_frequencies.get(doc_id, Counter())[term]

    def get_idf(self, term):
        # Return inverse document frequency of `term`
        total_docs = len(self.docmap)
        doc_freq = len(self.index.get(term, []))
        return math.log((total_docs + 1) / (doc_freq + 1))
    
    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]

            text = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id,text)

            self.docmap[doc_id] = movie
    
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(
                self.index,
                f
            )
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(
                    self.docmap,
                    f
                )
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(
                    self.term_frequencies,
                    f
                )

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)