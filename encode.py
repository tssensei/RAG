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

if __name__ == "__main__":
    try:
        encoder = QueryEncoder()
        test_query = "What is Cornell's add/drop policy?"
        vector = encoder.encode(test_query)
        
        print(f"\nTest Query: '{test_query}'")
        print(f"Generated Vector Shape: {vector.shape}")
        print(f"First 5 values: {vector[:5]}")
        
        if vector.shape[0] == 768:
            print("\nSUCCESS: Vector dimension is correct (768).")
        else:
            print(f"\nFAILED: Expected dimension 768, got {vector.shape[0]}")
            
    except Exception as e:
        print(f"\nError during testing: {e}")