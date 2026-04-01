import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

class RAGEvaluator:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self):
        """
        Loads the generated output data, runs RAGAS evaluation,
        and saves the result to a CSV file.
        """
        input_path = f"./code/evaluate/output/{self.name}_output/{self.name}_output.json"
        output_dir = f"./code/evaluate/output/{self.name}_output"
        output_path = f"{output_dir}/{self.name}_output_result.csv"

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        # Load the JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert dictionary to HuggingFace Dataset format expected by RAGAS
        # It typically expects keys: 'question', 'answer', 'contexts', 'ground_truth'
        dataset = Dataset.from_dict(data)

        # Define the metrics to evaluate
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

        print(f"Starting evaluation for: {self.name}...")
        
        # Initialize Embedding model
        embedding_model = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

        # Run the evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=embedding_model,
        )

        # Convert result to a pandas DataFrame
        df = result.to_pandas()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the result to a CSV file
        df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Results saved to: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # evaluator = RAGEvaluator("rai_curriculum")
    # evaluator.evaluate()
    pass