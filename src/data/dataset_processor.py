import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List

def process_rick_morty_data(csv_path: str, num_context: int = 7):
    """Process the Rick and Morty dataset."""
    df = pd.read_csv(csv_path)
    
    # Create context pairs
    contexted = []
    for i in range(num_context, len(df['line'])):
        row = []
        prev = i - 1 - num_context
        for j in range(i, prev, -1):
            row.append(df['line'][j])
        contexted.append(row)
    
    # Create DataFrame with context
    columns = ['response', 'context'] + [f'context/{i}' for i in range(num_context-1)]
    processed_df = pd.DataFrame.from_records(contexted, columns=columns)
    return processed_df

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):
        self.examples = []
        
        for _, row in df.iterrows():
            conv = self.construct_conv(row, tokenizer)
            if len(conv) <= block_size:
                self.examples.append(conv)

    def construct_conv(self, row, tokenizer, eos=True):
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] 
                            for x in row if isinstance(x, str)]))
        conv = flatten(conv)
        return conv

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)