import pandas as pd
import numpy as np
import os
import json
import re
from tqdm import tqdm
import logging
from src.config.improved_training_config import ImprovedTrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogueProcessor:
    def __init__(self, config):
        self.config = config
        self.rick_patterns = [
            r'\*burp\*',
            r'Wubba Lubba Dub Dub',
            r'Morty',
            r'dimension',
            r'portal',
            r'science'
        ]
        
    def clean_text(self, text):
        """Clean and normalize text"""
        text = str(text)
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        return text
    
    def is_valid_response(self, text):
        """Check if response meets quality criteria"""
        if not isinstance(text, str):
            return False
            
        # Length checks
        if len(text.split()) < self.config.min_response_length:
            return False
        if len(text.split()) > self.config.max_response_length:
            return False
            
        # Must contain actual words
        if not re.search(r'[a-zA-Z]{2,}', text):
            return False
            
        return True
    
    def is_rick_like(self, text):
        """Check if text contains Rick-like patterns"""
        text = text.lower()
        return any(re.search(pattern.lower(), text) for pattern in self.rick_patterns)
    
    def create_conversation_pairs(self, df):
        """Create context-response pairs from dialogue"""
        conversation_pairs = []
        
        for i in tqdm(range(len(df)), desc="Creating conversation pairs"):
            if 'Rick' not in str(df.iloc[i]['name']):
                continue
                
            response = self.clean_text(df.iloc[i]['line'])
            if not self.is_valid_response(response):
                continue
                
            # Get previous context
            context = []
            for j in range(max(0, i-3), i):
                speaker = str(df.iloc[j]['name'])
                text = self.clean_text(df.iloc[j]['line'])
                if self.is_valid_response(text):
                    context.append({
                        'speaker': speaker,
                        'text': text
                    })
            
            if context:  # Only add if we have context
                conversation_pairs.append({
                    'response': response,
                    'context': context,
                    'is_rick_like': self.is_rick_like(response)
                })
        
        return conversation_pairs

def process_data(input_file, output_dir):
    """Main data processing function"""
    config = ImprovedTrainingConfig()
    processor = DialogueProcessor(config)
    
    logger.info("Loading data...")
    df = pd.read_csv(input_file)
    
    logger.info("Processing dialogues...")
    conversation_pairs = processor.create_conversation_pairs(df)
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(conversation_pairs)
    
    # Filter for Rick-like responses
    rick_like_df = processed_df[processed_df['is_rick_like']]
    logger.info(f"Found {len(rick_like_df)} Rick-like dialogues out of {len(processed_df)} total")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    output_file = os.path.join(output_dir, 'improved_training_data.json')
    rick_like_df.to_json(output_file, orient='records', lines=True)
    logger.info(f"Saved processed data to {output_file}")
    
    # Print example
    print("\nExample processed dialogue:")
    example = rick_like_df.iloc[0]
    print(f"Response: {example['response']}")
    print("Context:")
    for ctx in example['context']:
        print(f"- {ctx['speaker']}: {ctx['text']}")
    
    return rick_like_df

if __name__ == "__main__":
    input_file = "data/raw/RickAndMortyScripts.csv"
    output_dir = "data/processed"
    
    try:
        processed_data = process_data(input_file, output_dir)
        print(f"\nSuccessfully processed {len(processed_data)} dialogues")
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise