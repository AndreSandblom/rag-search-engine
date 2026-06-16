import math
import os
import pickle
from collections import defaultdict, Counter

from .text_process import tokenize
from .search_utils import CACHE_PATH, load_movies,BM_25B,BM_25K1,DEFAULT_SEARCH_LIMIT


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = dict()
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id,text):
        tokens = tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        # update per-document term frequency counts
        self.term_frequencies[doc_id].update(tokens)
        # update document length
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        return total_length / num_docs if num_docs > 0 else 0

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
    
    def get_bm25_idf(self, term:str) -> float:
        total_docs = len(self.docmap)
        doc_freq = len(self.index.get(term, []))
        return math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM_25K1,b=BM_25B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length <= 0:
            avg_doc_length = 1.0
        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component
    
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    def bm25_search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        query_tokens = tokenize(query)
        scores = defaultdict(float)

        for doc_id in self.docmap:
            for token in query_tokens:
                bm25_score = self.bm25(doc_id, token)
                scores[doc_id] += bm25_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            movie = dict(self.docmap[doc_id])
            movie["score"] = score
            results.append(movie)
        return results

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
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(
                    self.doc_lengths,
                    f
                )

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)

        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)