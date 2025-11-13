"""
Converts MS MARCO documents into embeddings using BGE encoder
"""

import json
from FlagEmbedding import FlagModel
from tqdm import tqdm


def load_documents(file_path):
    print(f"Loading documents from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents")
    return documents

def encode_documents(documents, model, batch_size=32):
    print(f"Encoding {len(documents)} documents into embeddings...")
    print(f"Using batch size: {batch_size}")

    results = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding batches"):
        batch = documents[i:i + batch_size]
        texts = [doc['text'] for doc in batch]
        embeddings = model.encode(texts)
        for j, doc in enumerate(batch):
            results.append({
                'id': doc['id'],
                'text': doc['text'],
                'embedding': embeddings[j].tolist()
            })

    print(f"Successfully encoded {len(results)} documents")
    return results

def save_preprocessed_data(data, output_path):
    print(f"Saving preprocessed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def main():
    # INPUT_FILE = 'data_chunked/documents.json'
    INPUT_FILE = 'data_chunked/queries.json'
    # OUTPUT_FILE = 'chunked_output/preprocessed_documents.json'
    OUTPUT_FILE = 'chunked_output/preprocessed_queries.json'
    MODEL_NAME = 'BAAI/bge-base-en-v1.5'
    BATCH_SIZE = 32

    documents = load_documents(INPUT_FILE)

    print(f"\nInitializing BGE encoder model: {MODEL_NAME}")
    model = FlagModel(MODEL_NAME, use_fp16=False) 
    print("Model loaded successfully")

    print("\nStarting encoding process...")
    preprocessed_data = encode_documents(documents, model, batch_size=BATCH_SIZE)

    if preprocessed_data:
        embedding_dim = len(preprocessed_data[0]['embedding'])
        print(f"\nEmbedding dimension: {embedding_dim}")
        assert embedding_dim == 768, f"Expected 768 dimensions, got {embedding_dim}"

    print()
    save_preprocessed_data(preprocessed_data, OUTPUT_FILE)

    print("Preprocessing complete!")
    print(f"Total documents processed: {len(preprocessed_data)}")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
