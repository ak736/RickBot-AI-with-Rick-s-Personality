import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import random
from src.config.improved_training_config import ImprovedTrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRickBot:
    def __init__(self, model_path="models/rickbot-improved/final"):
        logger.info("Initializing Improved RickBot...")
        self.config = ImprovedTrainingConfig()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Generation parameters
        self.max_length = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.2
        
        # Rick-specific responses
        self.rick_catchphrases = [
            "Wubba Lubba Dub Dub!",
            "*burp* Listen, Morty...",
            "In this dimension,",
            "That's the waaaaay the news goes!",
            "And that's the science behind it, Morty!"
        ]
        
    def format_prompt(self, user_input):
        """Format the input with proper context"""
        return f"Morty: {user_input}\nRick:"
    
    def enhance_response(self, response):
        """Add Rick-like characteristics if needed"""
        if not response or len(response.strip()) < 10:
            return random.choice(self.rick_catchphrases)
            
        if "*burp*" not in response and random.random() < 0.3:
            response = f"*burp* {response}"
            
        return response

    def generate_response(self, user_input):
        try:
            # Format the prompt
            prompt = self.format_prompt(user_input)
            
            # Encode input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            # Enhance response
            response = self.enhance_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "*burp* Something went wrong in the multiverse!"
    
    def get_scientific_response(self, user_input):
        """Handle science-related questions"""
        science_prefixes = [
            "Let me break down the science for you, Morty.",
            "*burp* According to my research across infinite dimensions,",
            "The science is simple, even for your tiny brain:",
            "Through my experiments, I've discovered that"
        ]
        
        response = self.generate_response(user_input)
        return f"{random.choice(science_prefixes)} {response}"

def main():
    print("\nInitializing Improved RickBot... *burp*")
    rickbot = ImprovedRickBot()
    
    print("""
    RickBot: Wubba Lubba Dub Dub! I'm the smartest bot in the multiverse! *burp*
    
    Commands:
    - Type 'quit' to exit
    - Type 'science' before your question for scientific explanations
    - Just type normally for regular conversation
    
    Example: 'science how does gravity work?' or 'what do you think about Morty?'
    """)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nRickBot: Peace out, you tiny speck of cosmic dust! *burp*")
            break
            
        if not user_input:
            continue
            
        if user_input.lower().startswith('science'):
            response = rickbot.get_scientific_response(user_input[7:].strip())
        else:
            response = rickbot.generate_response(user_input)
            
        print(f"\nRickBot: {response}")

if __name__ == "__main__":
    main()