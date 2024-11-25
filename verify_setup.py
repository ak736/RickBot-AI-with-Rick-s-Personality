import os
import pandas as pd

def verify_project_setup():
    print("Verifying project setup...")
    
    # Check CSV file
    csv_path = os.path.join('data', 'raw', 'RickAndMortyScripts.csv')
    if os.path.exists(csv_path):
        print(f"✓ Found dataset at {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset loaded successfully with {len(df)} rows")
        print("\nFirst few rows of the dataset:")
        print(df.head())
        print("\nColumns in the dataset:", df.columns.tolist())
    else:
        print(f"✗ Dataset not found at {csv_path}")
    
    return df if os.path.exists(csv_path) else None

if __name__ == "__main__":
    df = verify_project_setup()