import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path="models/rickbot-small"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Download NLTK data
        try:
            nltk.download('punkt')
        except:
            pass

    def generate_response(self, context):
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=100,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.7,
            temperature=0.8
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def evaluate_model(self, test_data_path="data/processed/val.csv", num_samples=100):
        df = pd.read_csv(test_data_path)
        if len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=42)

        results = {
            'bleu_scores': [],
            'response_lengths': [],
            'unique_words': []
        }

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            context = row['context_0']  # Using first context as input
            actual_response = row['response']
            generated_response = self.generate_response(context)
            
            # Calculate BLEU score
            reference = nltk.word_tokenize(actual_response.lower())
            candidate = nltk.word_tokenize(generated_response.lower())
            bleu_score = sentence_bleu([reference], candidate)
            results['bleu_scores'].append(bleu_score)
            
            # Response statistics
            results['response_lengths'].append(len(candidate))
            results['unique_words'].append(len(set(candidate)))

        return self._compute_metrics(results)

    def _compute_metrics(self, results):
        metrics = {
            'avg_bleu_score': np.mean(results['bleu_scores']),
            'avg_response_length': np.mean(results['response_lengths']),
            'avg_unique_words': np.mean(results['unique_words']),
            'diversity_ratio': np.mean(results['unique_words']) / np.mean(results['response_lengths'])
        }
        return metrics

def main():
    logger.info("Starting model evaluation...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model()
    
    print("\nEvaluation Results:")
    print(f"Average BLEU Score: {metrics['avg_bleu_score']:.4f}")
    print(f"Average Response Length: {metrics['avg_response_length']:.2f} words")
    print(f"Average Unique Words: {metrics['avg_unique_words']:.2f}")
    print(f"Diversity Ratio: {metrics['diversity_ratio']:.4f}")

if __name__ == "__main__":
    main()