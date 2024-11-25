import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RickBot:
    def __init__(self, model_path="models/rickbot-improved"):
        logger.info("Initializing RickBot...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Set generation parameters
        self.max_new_tokens = 50
        self.temperature = 0.8
        self.top_p = 0.92
        self.top_k = 50
        self.repetition_penalty = 1.2
        
        # Rick-specific phrases for fallback and enhancements
        self.rick_phrases = [
            "*burp* Listen, Morty...",
            "Wubba lubba dub dub!",
            "In this dimension",
            "Let me tell you something about",
            "You know what happens when",
            "*burp* Here's the thing..."
        ]
        
    def enhance_response(self, response):
        """Add Rick-like characteristics if they're missing"""
        if not response:
            return random.choice(self.rick_phrases)
            
        # Add burp if not present
        if "*burp*" not in response and random.random() < 0.3:
            response = f"*burp* {response}"
            
        # Add common Rick phrases if response is too short
        if len(response.split()) < 5:
            response = f"{random.choice(self.rick_phrases)} {response}"
            
        return response

    def generate_response(self, user_input):
        try:
            # Create prompt with context
            prompt = f"Morty: {user_input}\nRick:"
            
            # Encode the input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and clean up the response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            # Enhance the response with Rick characteristics
            response = self.enhance_response(response)
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "*burp* Something went wrong in this dimension, Morty!"
    
    def get_scientific_response(self, user_input):
        """Special handling for science-related questions"""
        science_prefix = random.choice([
            "Let me break down this science stuff for you, Morty.",
            "*burp* The science behind this is actually pretty simple.",
            "In infinite dimensions, here's how this works:",
            "Only a real scientist would understand this, but..."
        ])
        
        response = self.generate_response(user_input)
        return f"{science_prefix} {response}"

def main():
    print("Initializing RickBot... *burp*")
    rickbot = RickBot()
    
    # Introduction message
    intro = """
    RickBot: Wubba Lubba Dub Dub! *burp* What's on your tiny mind, Morty?
    
    Commands:
    - Type 'quit' to exit
    - Type 'reset' to clear chat history
    - Type 'science' before your question for science-related responses
    """
    print(intro)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nRickBot: Peace out, you tiny speck in the infinite cosmos! *burp*")
            break
            
        if not user_input:
            continue
            
        # Handle different types of inputs
        if user_input.lower().startswith('science'):
            response = rickbot.get_scientific_response(user_input[7:].strip())
        else:
            response = rickbot.generate_response(user_input)
            
        print(f"\nRickBot: {response}")

if __name__ == "__main__":
    main()