from code.model.get_response import get_response
from code.service.chroma.chroma_retrieve import ChromaRetrieve
from config import DATA_NAME

class Pipeline:
    def __init__(self):
        self.name = DATA_NAME

    
    def ask(self, query):
        print("Starting RAG pipeline...")
        #User Input
        # query = input("Ask your question here: ")
        
        #RAG 1st Component: Retrieval
        chroma = ChromaRetrieve()
        results = chroma.retrieve(query)
        # print("\n\n\n", results)
        system_prompt = f"""
        You are a helpful assistant, you will answer the question from the information provided. 
        You don't use your internal knowledge. If you don't know the answer, just say Sorry, My knowledge base does not contain the answer to this question, and don't make things up

        Information provided:
        {str(results['documents'])}
        """

        # print(system_prompt)
        print("="*30+"Response"+"="*30)
        response = get_response(query, system_prompt)
        print(response)
        print("="*30+"Response"+"="*30)
        print("RAG pipeline completed.")
        return response
        
    def ask_with_context(self, query):
        print("Starting RAG pipeline with context...")
        chroma = ChromaRetrieve()
        results = chroma.retrieve(query)
        
        system_prompt = f"""
        You are a helpful assistant, you will answer the question from the information provided. 
        You don't use your internal knowledge. If you don't know the answer, just say Sorry, My knowledge base does not contain the answer to this question, and don't make things up

        Information provided:
        {str(results['documents'])}
        """

        print("="*30+"Response"+"="*30)
        response = get_response(query, system_prompt)
        print(response)
        print("="*30+"Response"+"="*30)
        print("RAG pipeline completed.")
        
        # results['documents'] is usually a list of lists of strings in chroma
        contexts = results.get('documents', [[]])[0] if results.get('documents') else []
        return response, contexts
