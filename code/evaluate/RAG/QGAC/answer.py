from config import DATA_NAME, CRITIC_MODEL, EMBEDDING_MODEL, MODEL_NAME
import json
import os
from datasets import Dataset
from code.pipeline import Pipeline


class Answer:
    def __init__(self, QA_data):
        self.name = DATA_NAME
        self.QGAC_name = f"QGAC__{DATA_NAME}__{EMBEDDING_MODEL}__{MODEL_NAME}.json"
        self.output_path = f"./code/evaluate/RAG/RAG_evaluation_file/{CRITIC_MODEL}/{DATA_NAME}/{EMBEDDING_MODEL}/{MODEL_NAME}/{self.QGAC_name}"
        self.QA_data = QA_data

        print(f"\n\n\nAnswering {self.name} dataset")
        print(f"Total QA Data: {len(self.QA_data)}")

    def generate_answer(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        pipeline = Pipeline()
        
        # Define a mapping function to process each row
        def get_answer(example):
            answer, contexts = pipeline.ask_with_context(example["question"])
            example["answer"] = answer
            example["contexts"] = contexts
            print(example["answer"])
            return example
        # Apply the mapping function to the dataset
        self.QA_output = self.QA_data.map(get_answer)
        
        self.save_output()
        return self.QA_output
    
    def save_output(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            # to_dict() works perfectly here on a Hugging Face Dataset
            json.dump(self.QA_output.to_dict(), f, ensure_ascii=False, indent=4)
        print(f"Answer saved to {self.output_path}")
    
    

if __name__ == "__main__":
    qa_data = [
        {"question": "What is the capital of France?", "answer": "", "contexts": ["Paris is the capital of France."], "ground_truth": "Paris"},
        {"question": "What is the largest city in France?", "answer": "", "contexts": ["Paris is the largest city in France."], "ground_truth": "Paris"},
        {"question": "What is the smallest city in France?", "answer": "", "contexts": ["Paris is the smallest city in France."], "ground_truth": "Paris"},
    ]
    answer = Answer(qa_data)
    answer.generate_answer()
   
