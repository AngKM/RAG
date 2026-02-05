from code.model.model_openai import Agent

class Pipeline:
    def __init__(self):
        self.agent = Agent()
        self.DATA_PATH = r"data"
        self.CHROMA_PATH = r"chroma_db"
    
    def run(self):
        print("Starting RAG pipeline...")
        print(self.agent.get_response("What is the capital of France?"))
        print("RAG pipeline completed.")
