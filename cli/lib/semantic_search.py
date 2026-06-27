import json
import re

from sentence_transformers import SentenceTransformer
from .search_utils import load_movies
import numpy as np

MODEL = 'all-MiniLM-L6-v2'
MAX_SEQ_LENGTH = 256

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def verify_model():
    try:
        model = SentenceTransformer(MODEL)
        print(f"Model loaded: {MODEL}")
        print(f"Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunking(text: str,max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+",text)
    sentences = [s for s in sentences if s.strip()]
    
    grouped_chunks = []
    step = max_chunk_size - overlap
    start = 0
    last_end = 0

    while start + max_chunk_size <= len(sentences):
        end = start + max_chunk_size
        chunk = " ".join(
            sentences[start:end]
        )
        grouped_chunks.append(chunk)
        last_end = end
        start += step
    if last_end < len(sentences):
        tail_start = max(last_end - overlap, 0)
        grouped_chunks.append(" ".join(sentences[tail_start:]))
    return grouped_chunks

def embed_chunks():
    movies = load_movies()
    semantic_search = ChunkedSemanticSearch()
    embeddings = semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

class SemanticSearch:
    def __init__(self,model_name: str = MODEL):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {} 

    def generate_embedding(self,text):
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.string_representations = []
        for doc in self.documents:
            doc_id = doc['id']
            value = doc
            self.document_map[doc_id] = value
            string_representation = f"{doc['title']}: {doc['description']}"
            self.string_representations.append(string_representation)
        self.embeddings = self.model.encode(self.string_representations,show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            doc_id = doc['id']
            value = doc
            self.document_map[doc_id] = value
        try:
            self.embeddings = np.load('cache/movie_embeddings.npy')
            print("Loaded embeddings from cache.")
            if self.embeddings.shape[0] == len(self.documents):
                return self.embeddings
            else:
                print("Cache size mismatch. Rebuilding embeddings...")
                return self.build_embeddings(documents)
        except FileNotFoundError:
            print("Cache not found. Building embeddings...")
            return self.build_embeddings(documents)

    def search(self, query, limit) -> list[str]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores_and_docs = []
        for idx, doc in enumerate(self.documents):
            doc_embedding = self.embeddings[idx]
            score = cosine_similarity(query_embedding, doc_embedding)
            scores_and_docs.append((score, doc))
        scores_and_docs.sort(key=lambda x: x[0], reverse=True)
        top_results = []
        for score, doc in scores_and_docs[:limit]:
            result_dict = {
                "score": score,
                "title": doc['title'],
                "description": doc['description']
            }
            top_results.append(result_dict)

        return top_results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents:list[dict]) -> np.ndarray:
        self.documents = documents
        self.chunk_metadata = []
        self.document_map = {}

        for doc in self.documents:
            doc_id = doc['id']
            value = doc
            self.document_map[doc_id] = value
        
        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(self.documents):
            if not doc["description"]:
                continue
            
            chunks = semantic_chunking(doc["description"], max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save('cache/chunk_embeddings.npy', self.chunk_embeddings)
        with open('cache/chunk_metadata.json', 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            doc_id = doc['id']
            value = doc
            self.document_map[doc_id] = value
        try:
            self.chunk_embeddings = np.load('cache/chunk_embeddings.npy')
            with open('cache/chunk_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
                total_chunks = metadata["total_chunks"]
            print("Loaded chunk embeddings and metadata from cache.")
            if total_chunks == len(self.chunk_metadata):
                return self.chunk_embeddings
            else:
                print("Cache size mismatch. Rebuilding chunk embeddings...")
                return self.build_chunk_embeddings(documents)
        except FileNotFoundError:
            print("Cache not found. Building chunk embeddings...")
            return self.build_chunk_embeddings(documents)