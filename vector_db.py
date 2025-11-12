"""
Vector Database using FAISS 
Builds and searches a vector index for semantic document retrieval
"""

import json
import numpy as np
import faiss
from typing import Tuple, List, Dict

class VectorDatabase:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = None
        self.documents = []  # document metadata (id, text)
        self.embeddings = None

    def load_documents(self, file_path):
        """Load preprocessed documents with embeddings from JSON file"""
        print(f"Loading preprocessed documents from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} documents")
        self.embeddings = np.array([doc['embedding'] for doc in data], dtype=np.float32)
        self.documents = [{'id': doc['id'], 'text': doc['text']} for doc in data]

        assert self.embeddings.shape[1] == self.dimension, \
            f"Expected {self.dimension} dimensions, got {self.embeddings.shape[1]}"
        print(f"Embeddings shape: {self.embeddings.shape}")

    def build_index(self):
        """Build FAISS index using L2 (Euclidean) distance"""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_documents() first.")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=5):
        """
        Search for k nearest neighbors

        Args:
            query_embedding: Query vector (shape: (768,) or (1, 768))
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (distances, indices):
                - distances: numpy array of shape (1, k) with L2 distances
                - indices: numpy array of shape (1, k) with document indices
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        distances, indices = self.index.search(query, k)
        return distances, indices

    def get_documents_by_indices(self, indices):
        results = []
        for idx in indices.flatten():
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
            else:
                results.append({'id': -1, 'text': 'Invalid index'})
        return results


def main():
    PREPROCESSED_FILE = 'preprocessed_documents.json'

    db = VectorDatabase(dimension=768)
    db.load_documents(PREPROCESSED_FILE)
    db.build_index()

    # document should match itself with distance ~0
    print("-"*60)
    print("Test 1: Self-Similarity Search")
    print()
    test_doc_idx = 42
    print(f"Document ID: {db.documents[test_doc_idx]['id']}")
    print(f"Document text: {db.documents[test_doc_idx]['text'][:100]}...")

    # Use its own embedding as query
    query_embedding = db.embeddings[test_doc_idx]
    distances, indices = db.search(query_embedding, k=5)

    print(f"\nTop 5 search results:")
    print(f"Distances: {distances[0]}")
    print(f"Indices:   {indices[0]}")

    if indices[0][0] == test_doc_idx and distances[0][0] < 0.001:
        print("\nSUCCESS: Document correctly matched itself with distance ≈ 0")
    else:
        print("\nWARNING: Self-match test failed")
    print()

if __name__ == "__main__":
    main()
