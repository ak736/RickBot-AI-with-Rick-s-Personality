# src/models/chat_with_improved_rick.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRickBot:
    def __init__(self, model_path=None):
        logger.info("Initializing Improved RickBot...")
        # Use DialoGPT model instead of local files
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Rick-specific parameters
        self.max_length = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.2

    def generate_response(self, user_input):
        try:
            inputs = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
            inputs = inputs.to(self.device)
            
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add Rick's style
            if not response.startswith('*burp*'):
                response = f'*burp* {response}'
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "*burp* Something went wrong in dimension C-137!"

    def get_scientific_response(self, question):
        try:
            # Add scientific context
            scientific_input = f"Explain scientifically: {question}"
            return self.generate_response(scientific_input)
            
        except Exception as e:
            logger.error(f"Error generating scientific response: {str(e)}")
            return "*burp* Even my genius has limits, Morty!"