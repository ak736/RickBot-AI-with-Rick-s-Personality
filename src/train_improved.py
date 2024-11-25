import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json
import logging
import os
from tqdm import tqdm
import gc
from src.config.improved_training_config import ImprovedTrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRickDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        logger.info(f"Loading data from {data_path}")
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format conversation
        context = [f"{ctx['speaker']}: {ctx['text']}" for ctx in item['context']]
        conversation = context + [f"Rick: {item['response']}"]
        
        # Join with special tokens
        text = self.tokenizer.eos_token.join(conversation)
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

def train_improved_model():
    config = ImprovedTrainingConfig()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set padding token to EOS token")
    
    # Load model with updated config
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': list(config.special_tokens.values())}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer")
    
    model.to(device)
    
    # Load dataset
    train_dataset = ImprovedRickDataset(
        "data/processed/improved_training_data.json",
        tokenizer,
        config.max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Memory cleanup
            del outputs
            del loss
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Average loss for epoch {epoch + 1}: {avg_loss}")
        
        # Save checkpoint
        output_dir = f"models/rickbot-improved/checkpoint-{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Save final model
    final_output_dir = "models/rickbot-improved/final"
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    train_improved_model()