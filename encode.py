from FlagEmbedding import FlagModel
import numpy as np

class QueryEncoder:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        print(f"Initializing Query Encoder with model: {model_name}...")
        self.model = FlagModel(model_name, use_fp16=False)
        print("Encoder model loaded successfully.")

    def encode(self, query_text):
        """
        Encodes a single query string into a embedding vector.
        """
        embedding = self.model.encode(query_text)
            
        return embedding
