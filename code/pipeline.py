from code.model.model_openai import Agent
from code.service.chroma.chroma import Chroma

class Pipeline:
    def __init__(self):
        self.agent = Agent()
        self.DATA_PATH = r"data"
        self.CHROMA_PATH = r"./chroma_db"

    
    def run(self):
        print("Starting RAG pipeline...")
        query = input("Ask your question here: ")
        chroma = Chroma(self.CHROMA_PATH)
        results = chroma.collection.query(
            query_texts=[query],
            n_results=100
        )
        print("="*20+"Result"+"="*20)
        print(results["ids"])
        print("="*20+"System Prompt"+"="*20)
        system_prompt = f"""
        You are a helpful startup assistant, you will answer the question for young startups from the information provided. 
        You don't use your internal knowledge. If you don't know the answer, just say I don't know, and don't make things up
        
        Information provided:
        {str(results['documents'])}
        """
        # print(system_prompt)
        response = self.agent.get_response(query, system_prompt)
        print(response)
        print("RAG pipeline completed.")
