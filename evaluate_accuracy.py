import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
from ragas.metrics import answer_similarity, answer_correctness
from dotenv import load_dotenv

load_dotenv()

def evaluate_accuracy(file_path: str, model_name: str) -> pd.DataFrame:
    """
    Loads JSON data, runs RAGAS accuracy evaluation (answer_similarity and answer_correctness),
    and returns a DataFrame with the results.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset_dict = {
        "question": data.get("question", [""] * len(data.get("answer", []))),
        "answer": data.get("answer", []),
        "ground_truth": data.get("ground_truth", [])
    }
    
    if "contexts" in data:
        dataset_dict["contexts"] = data["contexts"]
        
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"Starting accuracy evaluation for: {model_name}...")
    
    embedding_model = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    metrics = [answer_similarity, answer_correctness]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=embedding_model,
    )
    
    df = result.to_pandas()
    df.insert(0, 'Model', model_name)
    
    return df

if __name__ == "__main__":
    gpt_path = "./code/evaluate/output/rai_curriculum_gpt_output/rai_curriculum_gpt_output.json"
    rag_path = "./code/evaluate/output/rai_curriculum_output/rai_curriculum_output.json"
    
    df_gpt = evaluate_accuracy(gpt_path, "ChatGPT")
    df_rag = evaluate_accuracy(rag_path, "RAG Model")
    
    dfs = []
    if not df_gpt.empty:
        dfs.append(df_gpt)
    if not df_rag.empty:
        dfs.append(df_rag)
        
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        output_path = "accuracy_comparison.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\nEvaluation complete. Results saved to: {output_path}")
    else:
        print("\nNo output dataframes were generated. Check file paths and data content.")
