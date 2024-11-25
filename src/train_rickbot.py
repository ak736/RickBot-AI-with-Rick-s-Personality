import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import os
import json
from tqdm import tqdm
import gc
from config.training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RickMortyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        response = row['response']
        context = json.loads(row['context'])
        
        # Format the conversation
        conversation = []
        for ctx in context:
            conversation.append(f"{ctx['speaker']}: {ctx['text']}")
        conversation.append(f"Rick: {response}")
        
        # Join all parts with separator
        text = self.tokenizer.eos_token.join(conversation)
        
        # Tokenize
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

def train():
    config = TrainingConfig()
    
    # Memory cleanup
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(config.device)
    
    # Load dataset
    train_dataset = RickMortyDataset(config.train_data_path, tokenizer, config.max_seq_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                progress_bar.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})
                
                if global_step % config.save_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
            
            # Memory cleanup
            del outputs
            del loss
            if step % 100 == 0:
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Average loss for epoch {epoch + 1}: {avg_epoch_loss}")
        
        # Save epoch checkpoint
        save_path = os.path.join(config.output_dir, f"epoch-{epoch + 1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    # Save final model
    logger.info("Saving final model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    train()