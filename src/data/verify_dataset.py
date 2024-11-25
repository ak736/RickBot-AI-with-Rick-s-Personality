import pandas as pd
from dataset_processor import process_rick_morty_data

def verify_dataset():
    # Load and process the dataset
    dataset_path = '../data/raw/RickAndMortyScripts.csv'
    
    # Load original dataset
    print("Loading original dataset...")
    raw_df = pd.read_csv(dataset_path)
    print(f"Original dataset shape: {raw_df.shape}")
    print("\nFirst few rows of original dataset:")
    print(raw_df.head())
    
    # Process dataset
    print("\nProcessing dataset...")
    processed_df = process_rick_morty_data(dataset_path)
    print(f"Processed dataset shape: {processed_df.shape}")
    print("\nFirst few rows of processed dataset:")
    print(processed_df.head())
    
    return raw_df, processed_df

if __name__ == "__main__":
    raw_df, processed_df = verify_dataset()