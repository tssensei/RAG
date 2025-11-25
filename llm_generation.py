from llama_cpp import Llama
import os

class LLMGenerator:
    def __init__(self, model_path, n_ctx=2048):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
        print("LLM loaded successfully.")

    def generate_answer(self, prompt, max_tokens=512):
        output = self.llm(
            prompt,
            max_tokens=max_tokens, 
            stop=["Question:", "\n\n"], 
            echo=False 
        )
        
        response_text = output['choices'][0]['text'].strip()
        return response_text
