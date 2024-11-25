import torch

class TrainingConfig:
    def __init__(self):
        # Model parameters
        self.model_name = "microsoft/DialoGPT-small"
        self.model_type = "gpt2"
        self.tokenizer_name = "microsoft/DialoGPT-small"
        
        # Training parameters optimized for better quality
        self.num_train_epochs = 5  # Increased epochs for better learning
        self.per_device_train_batch_size = 2  # For M2 Air
        self.per_device_eval_batch_size = 2
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-5  # Slightly lower for better convergence
        self.weight_decay = 0.01
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        
        # Paths
        self.train_data_path = "data/processed/filtered_training_data.csv"
        self.output_dir = "models/rickbot-improved"
        self.cache_dir = "models/cache"
        
        # Device configuration
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Memory optimization
        self.max_seq_length = 128
        self.eval_steps = 200
        self.save_steps = 500
        self.logging_steps = 50