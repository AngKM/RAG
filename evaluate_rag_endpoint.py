from code.evaluate.evaluate_rag import RAGEvaluator
from config import DATA_NAME

if __name__ == "__main__":
    evaluator = RAGEvaluator(DATA_NAME)
    evaluator.evaluate()