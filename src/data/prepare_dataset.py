import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_training_data():
    """
    Prepare the Rick and Morty dataset for training
    """
    logger.info("Starting data preparation...")
    
    # Load the dataset
    df = pd.read_csv('data/raw/RickAndMortyScripts.csv')
    logger.info(f"Loaded {len(df)} lines from dataset")
    
    # Create conversation contexts
    contexted = []
    n_context = 7
    
    logger.info("Creating conversation contexts...")
    for i in range(n_context, len(df['line'])):
        row = {}
        # Current line is the response
        row['response'] = df['line'][i]
        
        # Previous n_context lines are the context
        for j in range(n_context):
            row[f'context_{j}'] = df['line'][i-j-1]
            row[f'speaker_{j}'] = df['name'][i-j-1]
        
        contexted.append(row)
    
    # Create DataFrame from the collected contexts
    processed_df = pd.DataFrame(contexted)
    
    logger.info(f"Created {len(processed_df)} conversation contexts")
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Split into train and validation sets
    logger.info("Splitting into train and validation sets...")
    train_df, val_df = train_test_split(processed_df, test_size=0.1, random_state=42)
    
    # Save processed datasets
    logger.info("Saving processed datasets...")
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    
    logger.info(f"Saved {len(train_df)} training examples and {len(val_df)} validation examples")
    
    # Print a sample conversation
    logger.info("\nSample conversation:")
    sample = train_df.iloc[0]
    print(f"Response: {sample['response']}")
    for i in range(n_context):
        print(f"Context {i} ({sample[f'speaker_{i}']}): {sample[f'context_{i}']}")
    
    return train_df, val_df

if __name__ == "__main__":
    train_df, val_df = prepare_training_data()