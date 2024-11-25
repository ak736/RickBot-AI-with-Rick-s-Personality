class ImprovedTrainingConfig:
    def __init__(self):
        # Model parameters
        self.model_name = "microsoft/DialoGPT-medium"  # Using medium instead of small
        self.max_length = 128
        self.batch_size = 2
        self.num_epochs = 3
        self.learning_rate = 2e-5
        
        # Data parameters
        self.min_response_length = 4
        self.max_response_length = 100
        
        # Special tokens
        self.special_tokens = {
            'burp': '*burp*',
            'catchphrase': 'Wubba Lubba Dub Dub',
            'morty': 'Morty',
            'rick': 'Rick',
            'sep': '<|sep|>'
        }
        
        # Training paths
        self.output_dir = "models/rickbot-improved"
        self.data_dir = "data/processed"
        
        # Memory optimization
        self.gradient_accumulation_steps = 4  # Added to help with memory
        self.warmup_steps = 50  # Reduced warmup steps
        self.save_steps = 200  # Save less frequently