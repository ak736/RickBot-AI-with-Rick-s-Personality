import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_training_data(input_file, output_file):
    logger.info("Loading dataset...")
    # Load the data
    df = pd.read_csv(input_file)
    
    logger.info(f"Original dataset size: {len(df)} rows")
    
    # Filter Rick's lines only
    rick_df = df[df['name'].astype(str).str.contains('Rick', case=False, na=False)]
    logger.info(f"Found {len(rick_df)} lines from Rick")
    
    # Remove very short or very long responses
    rick_df = rick_df[
        (rick_df['line'].astype(str).str.len() > 10) & 
        (rick_df['line'].astype(str).str.len() < 200)
    ]
    logger.info(f"After length filtering: {len(rick_df)} lines")
    
    # Remove lines that don't seem like dialogue
    rick_df = rick_df[
        ~rick_df['line'].astype(str).str.contains(r'^\W+$') &  
        ~rick_df['line'].astype(str).str.contains(r'^[0-9\s]+$')
    ]
    logger.info(f"After dialogue filtering: {len(rick_df)} lines")
    
    # Add context from previous lines
    logger.info("Adding context to conversations...")
    processed_dialogues = []
    n_context = 3
    
    for i in range(n_context, len(df)):
        if isinstance(df.iloc[i]['name'], str) and 'Rick' in df.iloc[i]['name']:
            context = []
            # Get previous n_context lines
            for j in range(i-n_context, i):
                if j >= 0:  # Ensure we don't go below index 0
                    context.append({
                        'text': str(df.iloc[j]['line']),
                        'speaker': str(df.iloc[j]['name'])
                    })
            
            processed_dialogues.append({
                'response': str(df.iloc[i]['line']),
                'context': json.dumps(context)  # Serialize context as JSON string
            })
    
    logger.info(f"Created {len(processed_dialogues)} dialogue examples with context")
    
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to DataFrame and save
    processed_df = pd.DataFrame(processed_dialogues)
    processed_df.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")
    
    # Print examples
    print("\nExample processed dialogues:")
    for i, row in processed_df.head(3).iterrows():
        print(f"\nExample {i+1}:")
        print(f"Response: {row['response']}")
        print("Context:")
        contexts = json.loads(row['context'])
        for ctx in contexts:
            print(f"- {ctx['speaker']}: {ctx['text']}")
    
    return processed_df

if __name__ == "__main__":
    input_file = "data/raw/RickAndMortyScripts.csv"
    output_file = "data/processed/filtered_training_data.csv"
    
    try:
        filtered_data = filter_training_data(input_file, output_file)
        print(f"Successfully processed {len(filtered_data)} dialogue examples")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise