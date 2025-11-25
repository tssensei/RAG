import sys
from encode import QueryEncoder
from vector_db import VectorDatabase
from llm_generation import LLMGenerator

def create_augmented_prompt(query, retrieved_documents):
    prompt = f"Question: {query}\n"
    prompt += "Context from relevant documents:\n"
    
    for i, doc in enumerate(retrieved_documents):
        prompt += f"{doc['text']}\n\n"
        
    prompt += "Based on the context above, provide a detailed answer. Answer:"
    return prompt

def main():
    PREPROCESSED_FILE = 'chunked_output/preprocessed_documents.json'
    MODEL_PATH = "qwen2-7b-instruct-q4_0.gguf" 
    
    db = VectorDatabase()
    db.load_documents(PREPROCESSED_FILE)
    db.build_index()
    encoder = QueryEncoder()
    llm = LLMGenerator(MODEL_PATH)

    while True:
        print("\n" + "-"*50)
        try:
            user_query = input("Enter your question: ").strip()
        except EOFError:
            break
        if user_query.lower() in ['exit', 'quit']:
            break
            
        print("\nProcessing...")
        
        # Encode Query
        query_vector = encoder.encode(user_query)
        # Vector Search
        distances, indices = db.search(query_vector, k=3)
        # Retrieve Documents
        retrieved_docs = db.get_documents_by_indices(indices)
        # Augment Prompt
        full_prompt = create_augmented_prompt(user_query, retrieved_docs)
        print("\n")
        print("Augmented Prompt:\n")
        print(full_prompt)
        # LLM Generation
        print("Generating answer...")
        answer = llm.generate_answer(full_prompt)
            
        print("\nResponse:")
        print(answer)
            
if __name__ == "__main__":
    main()