# src/train_bot.py
import logging
import os
import torch
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from config.model_config import Args
from data.dataset_processor import process_rick_morty_data, ConversationDataset
from models.trainer import train
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # Initialize configuration
    args = Args()
    
    # Setup device
    device = torch.device(args.device)
    args.device = device
    
    # Load tokenizer and model
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    
    # Process dataset
    dataset_path = os.path.join('data', 'raw', 'RickAndMortyScripts.csv')
    df = process_rick_morty_data(dataset_path)
    
    # Split the dataset
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Create train dataset
    train_dataset = ConversationDataset(tokenizer, args, train_df)
    
    # Train the model
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(f" global_step = {global_step}, average loss = {tr_loss}")
    
    # Save the final model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()